from __future__ import annotations

import argparse
import csv
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..common.config import ensure_dir, load_yaml, select_section, set_seed, setup_device, write_yaml
from ..common.data import (
    IndexSubset,
    PairwiseLaggedDataset,
    apply_stride,
    infer_n_states,
    load_dataset,
    pair_labels_from_state,
    select_model_inputs,
    split_train_val,
    unordered_pairs,
)
from .losses import total_pairwise_committor_loss
from .model import PairwiseCommittorNet


@dataclass
class TrainState:
    best_val: float = float("inf")
    best_epoch: int = -1


class GpuPairwiseLaggedBatcher:
    def __init__(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
        pair_labels: torch.Tensor,
        idx_t: torch.Tensor,
        idx_tau: torch.Tensor,
        subset_indices: np.ndarray,
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool = False,
    ):
        self.features = features
        self.weights = weights
        self.pair_labels = pair_labels
        ids = torch.as_tensor(subset_indices, dtype=torch.long, device=idx_t.device)
        self.idx_t = idx_t[ids].contiguous()
        self.idx_tau = idx_tau[ids].contiguous()
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

    def __len__(self) -> int:
        n = int(self.idx_t.numel())
        if self.drop_last:
            return n // self.batch_size
        return int(np.ceil(n / max(1, self.batch_size)))

    def __iter__(self):
        n = int(self.idx_t.numel())
        if self.shuffle:
            order = torch.randperm(n, device=self.idx_t.device)
        else:
            order = torch.arange(n, device=self.idx_t.device)
        stop = (n // self.batch_size) * self.batch_size if self.drop_last else n
        for start in range(0, stop, self.batch_size):
            batch_order = order[start : start + self.batch_size]
            if batch_order.numel() == 0 or (self.drop_last and batch_order.numel() < self.batch_size):
                continue
            i = self.idx_t[batch_order]
            j = self.idx_tau[batch_order]
            yield {
                "z_t": self.features[i],
                "z_tau": self.features[j],
                "weight": self.weights[i],
                "pair_label_t": self.pair_labels[i],
                "pair_label_tau": self.pair_labels[j],
            }


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device_type: str,
    drop_last: bool,
    prefetch_factor: int = 2,
) -> DataLoader:
    kwargs: dict[str, Any] = {}
    if device_type == "cuda":
        kwargs["pin_memory"] = True
        if int(num_workers) > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        drop_last=bool(drop_last),
        **kwargs,
    )


def autocast_context(use_amp: bool, device: torch.device):
    if not (use_amp and device.type == "cuda"):
        return nullcontext()
    try:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(dtype=torch.float16)


def make_grad_scaler(use_amp: bool, device: torch.device):
    enabled = bool(use_amp and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    if all(value.device == device for value in batch.values()):
        return batch
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def run_epoch(
    model: PairwiseCommittorNet,
    loader,
    optimizer: torch.optim.Optimizer | None,
    scaler,
    device: torch.device,
    *,
    train: bool,
    loss_cfg: dict[str, Any],
    use_amp: bool,
) -> dict[str, float]:
    model.train(train)
    totals = {
        "total_loss": torch.zeros((), dtype=torch.float64, device=device),
        "dirichlet_loss": torch.zeros((), dtype=torch.float64, device=device),
        "endpoint_loss": torch.zeros((), dtype=torch.float64, device=device),
    }
    n_seen = 0
    iterator = tqdm(loader, desc=("train" if train else "val"), leave=False)
    for batch in iterator:
        batch = move_batch(batch, device)
        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with autocast_context(use_amp, device):
                q_t = model(batch["z_t"])
                q_tau = model(batch["z_tau"])
                losses = total_pairwise_committor_loss(
                    q_t=q_t,
                    q_tau=q_tau,
                    pair_label_t=batch["pair_label_t"],
                    pair_label_tau=batch["pair_label_tau"],
                    weights=batch["weight"],
                    lambda_dirichlet=float(loss_cfg.get("lambda_dirichlet", 1.0)),
                    lambda_endpoint=float(loss_cfg.get("lambda_endpoint", 100.0)),
                    weighted_dirichlet=bool(loss_cfg.get("weighted_dirichlet", True)),
                    weighted_endpoint=bool(loss_cfg.get("weighted_endpoint", False)),
                )
                loss = losses["total_loss"]
            if train and optimizer is not None:
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        with torch.no_grad():
            bs = int(batch["z_t"].shape[0])
            n_seen += bs
            for key in totals:
                totals[key] += losses[key].detach().double() * bs
            if not isinstance(loader, GpuPairwiseLaggedBatcher):
                iterator.set_postfix(loss=f"{float(loss.detach().cpu()):.3e}")
    denom = float(max(1, n_seen))
    return {key: float((value / denom).detach().cpu().item()) for key, value in totals.items()}


def append_history(path: str, epoch: int, train_stats: dict[str, float], val_stats: dict[str, float]) -> None:
    row = {"epoch": int(epoch)}
    for prefix, stats in (("train", train_stats), ("val", val_stats)):
        for key, value in stats.items():
            row[f"{prefix}_{key}"] = float(value)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_scripted_model(model: PairwiseCommittorNet, path: str, device: torch.device) -> None:
    model_cpu = model.to("cpu")
    torch.jit.script(model_cpu).save(path)
    model.to(device)


def make_checkpoint(
    model: PairwiseCommittorNet,
    config: dict[str, Any],
    input_meta: dict[str, Any],
    n_states: int,
    pairs: list[tuple[int, int]],
    state: TrainState,
    epoch: int,
    stats: dict[str, float] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_kwargs": model.model_kwargs(),
        "config": config,
        "model_input": input_meta,
        "n_states": int(n_states),
        "pairs": pairs,
        "best_epoch": int(state.best_epoch),
        "best_val": float(state.best_val),
        "epoch": int(epoch),
    }
    if stats is not None:
        payload["stats"] = {key: float(value) for key, value in stats.items()}
    return payload


def train_pairwise_committor(config: dict[str, Any]) -> dict[str, Any]:
    out_dir = ensure_dir(config.get("out_dir", "./pairwise_committor_train"))
    label = str(config.get("label", "pairwise_committor"))
    device = setup_device(config.get("device", "cuda:0"))
    seed = int(config.get("seed", 0))
    set_seed(seed)

    pack = apply_stride(load_dataset(config["dataset_path"]), int(config.get("dataset_stride", 1)))
    n_states = infer_n_states(pack, config.get("n_states", None))
    pairs = unordered_pairs(n_states)
    model_features, input_meta = select_model_inputs(pack, config)
    pair_labels = pair_labels_from_state(pack.state, n_states)
    n_frames, in_dim = model_features.shape

    lag = int(config.get("lag", 1))
    ds = PairwiseLaggedDataset(
        features=model_features,
        weights=pack.weights,
        pair_labels=pair_labels,
        lag=lag,
        traj_id=pack.traj_id,
        allow_cross_traj_pairs=bool(config.get("allow_cross_traj_pairs", False)),
        require_labeled=str(config.get("require_labeled", "both")),
    )
    tr_idx, va_idx = split_train_val(len(ds), float(config.get("val_ratio", 0.1)), seed=seed)

    batch_size = int(config.get("batch_size", 2048))
    drop_last = bool(config.get("drop_last", False))
    gpu_resident_data = bool(config.get("gpu_resident_data", device.type == "cuda"))
    if gpu_resident_data and device.type == "cuda":
        model_features = model_features.contiguous().to(device)
        weights_gpu = pack.weights.contiguous().to(device)
        pair_labels_gpu = pair_labels.contiguous().to(device)
        idx_t_gpu = ds.idx_t.contiguous().to(device)
        idx_tau_gpu = ds.idx_tau.contiguous().to(device)
        train_loader = GpuPairwiseLaggedBatcher(
            model_features,
            weights_gpu,
            pair_labels_gpu,
            idx_t_gpu,
            idx_tau_gpu,
            tr_idx,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
        )
        val_loader = GpuPairwiseLaggedBatcher(
            model_features,
            weights_gpu,
            pair_labels_gpu,
            idx_t_gpu,
            idx_tau_gpu,
            va_idx,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        data_residency = "gpu"
    else:
        num_workers = int(config.get("num_workers", 0))
        train_loader = make_loader(
            IndexSubset(ds, tr_idx),
            batch_size,
            True,
            num_workers,
            device.type,
            drop_last,
            prefetch_factor=int(config.get("prefetch_factor", 2)),
        )
        val_loader = make_loader(
            IndexSubset(ds, va_idx),
            batch_size,
            False,
            num_workers,
            device.type,
            False,
            prefetch_factor=int(config.get("prefetch_factor", 2)),
        )
        data_residency = "cpu"

    model_cfg = config.get("model", {})
    if not isinstance(model_cfg, dict):
        raise ValueError("model config must be a mapping.")
    model = PairwiseCommittorNet(
        in_dim=int(in_dim),
        n_pairs=len(pairs),
        hidden=model_cfg.get("hidden", [256, 256, 128]),
        activation=model_cfg.get("activation", "elu"),
        dropout=float(model_cfg.get("dropout", 0.0)),
        batch_norm=bool(model_cfg.get("batch_norm", False)),
        output_activation=model_cfg.get("output_activation", "sigmoid"),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-6)),
    )
    scaler = make_grad_scaler(bool(config.get("use_amp", True)), device)
    loss_cfg = dict(config.get("loss", {}))
    state = TrainState()
    best_state: dict[str, torch.Tensor] | None = None
    best_stats: dict[str, float] | None = None

    history_path = os.path.join(out_dir, f"{label}_history.csv")
    best_model_path = os.path.join(out_dir, f"{label}_best_model.pt")
    last_model_path = os.path.join(out_dir, f"{label}_last_model.pt")
    best_ckpt_path = os.path.join(out_dir, f"{label}_best_checkpoint.pt")
    last_ckpt_path = os.path.join(out_dir, f"{label}_last_checkpoint.pt")

    print(
        f"[DATA] frames={n_frames}, feature_dim={in_dim}, "
        f"model_input_space={input_meta['model_input_space']}, data_residency={data_residency}, "
        f"lag={lag}, pairs={len(ds)}, n_states={n_states}"
    )
    print(f"[MODEL] unordered pair columns={pairs}")

    epochs = int(config.get("epochs", 500))
    patience = int(config.get("patience", 30))
    use_amp = bool(config.get("use_amp", True))
    for epoch in range(1, epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, scaler, device, train=True, loss_cfg=loss_cfg, use_amp=use_amp)
        val_stats = run_epoch(model, val_loader, None, scaler, device, train=False, loss_cfg=loss_cfg, use_amp=use_amp)
        append_history(history_path, epoch, train_stats, val_stats)
        if val_stats["total_loss"] < state.best_val - 1e-10:
            state.best_val = float(val_stats["total_loss"])
            state.best_epoch = int(epoch)
            best_state = clone_state_dict(model)
            best_stats = val_stats
            torch.save(make_checkpoint(model, config, input_meta, n_states, pairs, state, epoch, val_stats), best_ckpt_path)
            print(f"[CHECKPOINT] Saved best checkpoint: {best_ckpt_path}")
        print(
            f"[Epoch {epoch:4d}] train={train_stats['total_loss']:.6e} "
            f"(dir={train_stats['dirichlet_loss']:.3e}, endpoint={train_stats['endpoint_loss']:.3e})  "
            f"val={val_stats['total_loss']:.6e} "
            f"(dir={val_stats['dirichlet_loss']:.3e}, endpoint={val_stats['endpoint_loss']:.3e})  "
            f"best={state.best_val:.6e}@{state.best_epoch}"
        )
        if epoch - state.best_epoch >= patience:
            print(f"[EARLY STOP] No validation improvement for {patience} epochs.")
            break

    if best_state is None:
        best_state = clone_state_dict(model)
        best_stats = val_stats
    live_state = clone_state_dict(model)
    model.load_state_dict(best_state)
    save_scripted_model(model, best_model_path, device)
    torch.save(make_checkpoint(model, config, input_meta, n_states, pairs, state, state.best_epoch, best_stats), best_ckpt_path)
    model.load_state_dict(live_state)
    save_scripted_model(model, last_model_path, device)
    torch.save(make_checkpoint(model, config, input_meta, n_states, pairs, state, epoch, val_stats), last_ckpt_path)

    summary = {
        "best_model": os.path.abspath(best_model_path),
        "last_model": os.path.abspath(last_model_path),
        "best_checkpoint": os.path.abspath(best_ckpt_path),
        "last_checkpoint": os.path.abspath(last_ckpt_path),
        "history_csv": os.path.abspath(history_path),
        "best_epoch": int(state.best_epoch),
        "best_val": float(state.best_val),
        "n_states": int(n_states),
        "pairs": [[int(i), int(j)] for i, j in pairs],
        "n_frames": int(n_frames),
        "feature_dim": int(in_dim),
        "model_input": input_meta,
        "data_residency": data_residency,
        "gpu_resident_data": bool(gpu_resident_data),
        "lag": int(lag),
        "dataset_path": os.path.abspath(str(config["dataset_path"])),
    }
    summary_path = os.path.join(out_dir, f"{label}_summary.yaml")
    write_yaml(summary, summary_path)
    print(f"[DONE] Best model: {best_model_path}")
    print(f"[DONE] Summary: {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal pair-wise committor model.")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()
    raw = load_yaml(args.config)
    cfg = select_section(raw, "PAIRWISE_COMMITTOR", "PAIRWISE_TRAIN", "TRAIN")
    train_pairwise_committor(cfg)


if __name__ == "__main__":
    main()
