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
    LaggedCommittorDataset,
    apply_stride,
    infer_n_states,
    load_dataset,
    select_model_inputs,
    split_train_val,
)
from ..common.flux import make_thresholds, resolve_ordered_pairs
from .losses import total_committor_loss
from .metrics import endpoint_boundary_accuracy, mean_entropy, normalization_error
from .model import NextHitCommittorNet


@dataclass
class TrainState:
    best_val: float = float("inf")
    best_epoch: int = -1


def clone_model_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def validate_state_labels(state: torch.Tensor, n_states: int, dataset_path: str) -> None:
    state_cpu = state.detach().cpu().long()
    labeled = state_cpu[state_cpu >= 0]
    if labeled.numel() == 0:
        return
    min_label = int(labeled.min().item())
    max_label = int(labeled.max().item())
    if min_label < 0 or max_label >= int(n_states):
        counts = torch.bincount(labeled, minlength=max(max_label + 1, int(n_states))).cpu().numpy()
        nonzero = [(idx, int(count)) for idx, count in enumerate(counts) if count]
        raise ValueError(
            f"Dataset labels in {dataset_path} are outside n_states={n_states}: "
            f"valid labeled range is [{min_label}, {max_label}]. "
            f"Set n_states >= {max_label + 1} or regenerate the relabeled dataset metadata. "
            f"Observed labeled counts: {nonzero}"
        )


class GpuLaggedBatcher:
    def __init__(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
        state: torch.Tensor,
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
        self.state = state
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
                "state_t": self.state[i],
                "state_tau": self.state[j],
            }


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device_type: str,
    drop_last: bool = False,
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


def make_grad_scaler(use_amp: bool, device: torch.device):
    enabled = bool(use_amp and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(use_amp: bool, device: torch.device):
    if not (use_amp and device.type == "cuda"):
        return nullcontext()
    try:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(dtype=torch.float16)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    if all(value.device == device for value in batch.values()):
        return batch
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _zero_pair_arrays(n_pairs: int, n_thresholds: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    J = torch.zeros((n_pairs, n_thresholds), dtype=torch.float64, device=device)
    V = torch.zeros((n_pairs,), dtype=torch.float64, device=device)
    return J, V


def run_epoch(
    model: NextHitCommittorNet,
    loader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    *,
    train: bool,
    pairs: list[tuple[int, int]],
    thresholds: torch.Tensor,
    loss_cfg: dict[str, Any],
    use_amp: bool,
) -> dict[str, Any]:
    model.train(train)

    totals = {
        "total_loss": torch.zeros((), dtype=torch.float64, device=device),
        "dirichlet_loss": torch.zeros((), dtype=torch.float64, device=device),
        "boundary_loss": torch.zeros((), dtype=torch.float64, device=device),
        "flux_loss": torch.zeros((), dtype=torch.float64, device=device),
        "boundary_accuracy": torch.zeros((), dtype=torch.float64, device=device),
        "mean_entropy": torch.zeros((), dtype=torch.float64, device=device),
        "normalization_error": torch.zeros((), dtype=torch.float64, device=device),
    }
    J_sum, V_sum = _zero_pair_arrays(len(pairs), thresholds.numel(), device)
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
                losses = total_committor_loss(
                    q_t=q_t,
                    q_tau=q_tau,
                    state_t=batch["state_t"],
                    state_tau=batch["state_tau"],
                    pairs=pairs,
                    thresholds=thresholds,
                    flux_eps=float(loss_cfg.get("flux_eps", 0.02)),
                    sample_weights=batch["weight"],
                    lambda_dir=float(loss_cfg.get("lambda_dir", 1.0)),
                    lambda_bc=float(loss_cfg.get("lambda_bc", 10.0)),
                    lambda_flux=float(loss_cfg.get("lambda_flux", 0.1)),
                    boundary_mode=str(loss_cfg.get("boundary_mode", "cross_entropy")),
                    weighted_dirichlet=bool(loss_cfg.get("weighted_dirichlet", True)),
                    weighted_boundary=bool(loss_cfg.get("weighted_boundary", False)),
                    weighted_flux=bool(loss_cfg.get("weighted_flux", True)),
                    tau=float(loss_cfg.get("tau", loss_cfg.get("lag", 1.0))),
                    scale_dirichlet_by_tau=bool(loss_cfg.get("scale_dirichlet_by_tau", False)),
                    scale_flux_by_tau=bool(loss_cfg.get("scale_flux_by_tau", False)),
                    flux_surface=str(loss_cfg.get("flux_surface", "qi_decrease")),
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
            totals["total_loss"] += losses["total_loss"].detach().double() * bs
            totals["dirichlet_loss"] += losses["dirichlet_loss"].detach().double() * bs
            totals["boundary_loss"] += losses["boundary_loss"].detach().double() * bs
            totals["flux_loss"] += losses["flux_loss"].detach().double() * bs
            totals["boundary_accuracy"] += endpoint_boundary_accuracy(
                q_t.detach(), q_tau.detach(), batch["state_t"], batch["state_tau"]
            ).double() * bs
            totals["mean_entropy"] += (0.5 * (mean_entropy(q_t.detach()) + mean_entropy(q_tau.detach()))).double() * bs
            totals["normalization_error"] += torch.maximum(
                normalization_error(q_t.detach()), normalization_error(q_tau.detach())
            ).double() * bs
            J_sum += losses["J"].detach().double() * bs
            V_sum += losses["flux_variance"].detach().double() * bs

            if not isinstance(loader, GpuLaggedBatcher):
                iterator.set_postfix(loss=f"{float(loss.detach().cpu()):.3e}")

    denom = float(max(1, n_seen))
    stats: dict[str, Any] = {key: float((value / denom).detach().cpu().item()) for key, value in totals.items()}
    J_mean = (J_sum / denom).detach().cpu().numpy()
    V_mean = (V_sum / denom).detach().cpu().numpy()
    stats["J"] = J_mean
    stats["flux_variance_by_pair"] = V_mean
    stats["n_samples"] = int(n_seen)
    return stats


def save_scripted_model(model: NextHitCommittorNet, path: str, device: torch.device) -> None:
    model_cpu = model.to("cpu")
    scripted = torch.jit.script(model_cpu)
    scripted.save(path)
    model.to(device)


def append_history(path: str, epoch: int, train_stats: dict[str, Any], val_stats: dict[str, Any]) -> None:
    scalar_keys = [
        "total_loss",
        "dirichlet_loss",
        "boundary_loss",
        "flux_loss",
        "boundary_accuracy",
        "mean_entropy",
        "normalization_error",
    ]
    row = {"epoch": epoch}
    for prefix, stats in (("train", train_stats), ("val", val_stats)):
        for key in scalar_keys:
            row[f"{prefix}_{key}"] = stats[key]
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_flux_snapshot(
    out_path: str,
    pairs: list[tuple[int, int]],
    thresholds: np.ndarray,
    J: np.ndarray,
    variance: np.ndarray,
) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["state_i", "state_j", "threshold", "J_ij", "threshold_variance"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p_idx, (i, j) in enumerate(pairs):
            for t_idx, c in enumerate(thresholds):
                writer.writerow(
                    {
                        "state_i": i,
                        "state_j": j,
                        "threshold": float(c),
                        "J_ij": float(J[p_idx, t_idx]),
                        "threshold_variance": float(variance[p_idx]),
                    }
                )


def make_checkpoint_payload(
    model: NextHitCommittorNet,
    config: dict[str, Any],
    n_states: int,
    pairs: list[tuple[int, int]],
    thresholds: torch.Tensor,
    input_meta: dict[str, Any],
    state: TrainState,
    *,
    epoch: int,
    train_stats: dict[str, Any] | None = None,
    val_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_kwargs": model.model_kwargs(),
        "config": config,
        "n_states": n_states,
        "pairs": pairs,
        "thresholds": thresholds.detach().cpu(),
        "epoch": int(epoch),
        "best_epoch": state.best_epoch,
        "best_val": state.best_val,
        "model_input": input_meta,
    }
    if train_stats is not None:
        payload["train_stats"] = {
            key: value for key, value in train_stats.items() if not isinstance(value, np.ndarray)
        }
    if val_stats is not None:
        payload["val_stats"] = {
            key: value for key, value in val_stats.items() if not isinstance(value, np.ndarray)
        }
    return payload


def train_next_hit_committor(config: dict[str, Any]) -> dict[str, Any]:
    out_dir = ensure_dir(config.get("out_dir", "./next_hit_train"))
    label = str(config.get("label", "next_hit_committor"))
    device = setup_device(config.get("device", "cuda:0"))
    seed = int(config.get("seed", 0))
    set_seed(seed)

    dataset_path = config["dataset_path"]
    pack = apply_stride(load_dataset(dataset_path), int(config.get("dataset_stride", 1)))
    n_states = infer_n_states(pack, config.get("n_states", None))
    validate_state_labels(pack.state, n_states, str(dataset_path))
    model_features, input_meta = select_model_inputs(pack, config)
    n_frames, in_dim = model_features.shape

    lag = int(config.get("lag", config.get("time_shift", 1)))
    require_labeled = str(config.get("require_labeled", "none"))
    ds = LaggedCommittorDataset(
        features=model_features,
        weights=pack.weights,
        state=pack.state,
        lag=lag,
        traj_id=pack.traj_id,
        allow_cross_traj_pairs=bool(config.get("allow_cross_traj_pairs", False)),
        require_labeled=require_labeled,
    )
    tr_idx, va_idx = split_train_val(len(ds), val_ratio=float(config.get("val_ratio", 0.1)), seed=seed)

    batch_size = int(config.get("batch_size", 2048))
    drop_last = bool(config.get("drop_last", False))
    gpu_resident_data = bool(config.get("gpu_resident_data", device.type == "cuda"))
    if gpu_resident_data and device.type == "cuda":
        model_features = model_features.contiguous().to(device)
        weights_gpu = pack.weights.contiguous().to(device)
        state_gpu = pack.state.contiguous().to(device)
        idx_t_gpu = ds.idx_t.contiguous().to(device)
        idx_tau_gpu = ds.idx_tau.contiguous().to(device)
        train_loader = GpuLaggedBatcher(
            model_features,
            weights_gpu,
            state_gpu,
            idx_t_gpu,
            idx_tau_gpu,
            tr_idx,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
        )
        val_loader = GpuLaggedBatcher(
            model_features,
            weights_gpu,
            state_gpu,
            idx_t_gpu,
            idx_tau_gpu,
            va_idx,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        data_residency = "gpu"
    else:
        train_ds = IndexSubset(ds, tr_idx)
        val_ds = IndexSubset(ds, va_idx)
        num_workers = int(config.get("num_workers", 0))
        train_loader = make_loader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            device_type=device.type,
            drop_last=drop_last,
            prefetch_factor=int(config.get("prefetch_factor", 2)),
        )
        val_loader = make_loader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            device_type=device.type,
            drop_last=False,
            prefetch_factor=int(config.get("prefetch_factor", 2)),
        )
        data_residency = "cpu"

    model_cfg = config.get("model", {})
    if not isinstance(model_cfg, dict):
        raise ValueError("model config must be a mapping.")
    hidden = model_cfg.get("hidden", config.get("hidden", [256, 256, 128]))
    model = NextHitCommittorNet(
        in_dim=int(in_dim),
        n_states=n_states,
        hidden=hidden,
        activation=model_cfg.get("activation", config.get("activation", "elu")),
        dropout=float(model_cfg.get("dropout", config.get("dropout", 0.0))),
        batch_norm=bool(model_cfg.get("batch_norm", config.get("batch_norm", False))),
        output_normalization=model_cfg.get("output_normalization", config.get("output_normalization", "softmax")),
        positive_eps=float(model_cfg.get("positive_eps", config.get("positive_eps", 1e-8))),
        use_softmax_output=model_cfg.get("use_softmax_output", config.get("use_softmax_output", None)),
    ).to(device)

    loss_cfg = dict(config.get("loss", {}))
    loss_cfg.setdefault("lag", lag)
    thresholds = make_thresholds(
        loss_cfg.get("thresholds", None),
        n_thresholds=int(loss_cfg.get("n_thresholds", 9)),
        start=float(loss_cfg.get("threshold_start", 0.1)),
        stop=float(loss_cfg.get("threshold_stop", 0.9)),
        device=device,
    )
    pairs = resolve_ordered_pairs(
        n_states,
        adjacency=loss_cfg.get("adjacency", loss_cfg.get("flux_pairs", None)),
        symmetric_adjacency=bool(loss_cfg.get("symmetric_adjacency", True)),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", config.get("learning_rate", 1e-3))),
        weight_decay=float(config.get("weight_decay", 1e-6)),
    )
    use_amp = bool(config.get("use_amp", True))
    scaler = make_grad_scaler(use_amp, device)

    epochs = int(config.get("epochs", 500))
    patience = int(config.get("patience", 30))
    history_path = os.path.join(out_dir, f"{label}_history.csv")
    best_model_path = os.path.join(out_dir, f"{label}_best_model.pt")
    last_model_path = os.path.join(out_dir, f"{label}_last_model.pt")
    best_ckpt_path = os.path.join(out_dir, f"{label}_best_checkpoint.pt")
    last_ckpt_path = os.path.join(out_dir, f"{label}_last_checkpoint.pt")
    flux_train_path = os.path.join(out_dir, f"{label}_train_flux_profiles.csv")
    flux_val_path = os.path.join(out_dir, f"{label}_val_flux_profiles.csv")

    state = TrainState()
    best_model_state: dict[str, torch.Tensor] | None = None
    best_train_stats: dict[str, Any] | None = None
    best_val_stats: dict[str, Any] | None = None
    print(
        f"[DATA] frames={n_frames}, feature_dim={in_dim}, "
        f"model_input_space={input_meta['model_input_space']}, data_residency={data_residency}, "
        f"lag={lag}, pairs={len(ds)}, n_states={n_states}"
    )
    print(
        f"[MODEL] output shape = (batch_size, {n_states}); "
        f"output_normalization = {model.output_normalization}"
    )
    print(f"[FLUX] ordered pairs = {pairs}")

    for epoch in range(1, epochs + 1):
        train_stats = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            train=True,
            pairs=pairs,
            thresholds=thresholds,
            loss_cfg=loss_cfg,
            use_amp=use_amp,
        )
        val_stats = run_epoch(
            model,
            val_loader,
            optimizer=None,
            scaler=scaler,
            device=device,
            train=False,
            pairs=pairs,
            thresholds=thresholds,
            loss_cfg=loss_cfg,
            use_amp=use_amp,
        )
        append_history(history_path, epoch, train_stats, val_stats)

        improved = val_stats["total_loss"] < state.best_val - 1e-10
        if improved:
            state.best_val = float(val_stats["total_loss"])
            state.best_epoch = epoch
            best_model_state = clone_model_state_dict(model)
            best_train_stats = train_stats
            best_val_stats = val_stats
            torch.save(
                make_checkpoint_payload(
                    model,
                    config,
                    n_states,
                    pairs,
                    thresholds,
                    input_meta,
                    state,
                    epoch=epoch,
                    train_stats=train_stats,
                    val_stats=val_stats,
                ),
                best_ckpt_path,
            )
            print(f"[CHECKPOINT] Saved best checkpoint: {best_ckpt_path}")

        print(
            f"[Epoch {epoch:4d}] "
            f"train={train_stats['total_loss']:.6e} "
            f"(dir={train_stats['dirichlet_loss']:.3e}, bc={train_stats['boundary_loss']:.3e}, "
            f"flux={train_stats['flux_loss']:.3e}, acc={train_stats['boundary_accuracy']:.3f}, "
            f"H={train_stats['mean_entropy']:.3f})  "
            f"val={val_stats['total_loss']:.6e} "
            f"(dir={val_stats['dirichlet_loss']:.3e}, bc={val_stats['boundary_loss']:.3e}, "
            f"flux={val_stats['flux_loss']:.3e}, acc={val_stats['boundary_accuracy']:.3f}, "
            f"H={val_stats['mean_entropy']:.3f})  "
            f"best={state.best_val:.6e}@{state.best_epoch}"
        )

        if epoch - state.best_epoch >= patience:
            print(f"[EARLY STOP] No validation improvement for {patience} epochs.")
            break

    if best_model_state is None:
        best_model_state = clone_model_state_dict(model)
        best_train_stats = train_stats
        best_val_stats = val_stats

    live_model_state = clone_model_state_dict(model)
    model.load_state_dict(best_model_state)
    save_scripted_model(model, best_model_path, device)
    torch.save(
        make_checkpoint_payload(
            model,
            config,
            n_states,
            pairs,
            thresholds,
            input_meta,
            state,
            epoch=state.best_epoch,
            train_stats=best_train_stats,
            val_stats=best_val_stats,
        ),
        best_ckpt_path,
    )
    if best_val_stats is not None:
        save_flux_snapshot(
            flux_val_path,
            pairs=pairs,
            thresholds=thresholds.detach().cpu().numpy(),
            J=best_val_stats["J"],
            variance=best_val_stats["flux_variance_by_pair"],
        )

    model.load_state_dict(live_model_state)
    save_scripted_model(model, last_model_path, device)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kwargs": model.model_kwargs(),
            "config": config,
            "n_states": n_states,
            "pairs": pairs,
            "thresholds": thresholds.detach().cpu(),
            "best_epoch": state.best_epoch,
            "best_val": state.best_val,
            "model_input": input_meta,
        },
        last_ckpt_path,
    )
    save_flux_snapshot(
        flux_train_path,
        pairs=pairs,
        thresholds=thresholds.detach().cpu().numpy(),
        J=train_stats["J"],
        variance=train_stats["flux_variance_by_pair"],
    )

    summary = {
        "best_model": os.path.abspath(best_model_path),
        "last_model": os.path.abspath(last_model_path),
        "best_checkpoint": os.path.abspath(best_ckpt_path),
        "last_checkpoint": os.path.abspath(last_ckpt_path),
        "history_csv": os.path.abspath(history_path),
        "best_epoch": int(state.best_epoch),
        "best_val": float(state.best_val),
        "n_frames": int(n_frames),
        "feature_dim": int(in_dim),
        "model_input": input_meta,
        "data_residency": data_residency,
        "gpu_resident_data": bool(gpu_resident_data),
        "n_states": int(n_states),
        "output_normalization": model.output_normalization,
        "lag": int(lag),
        "ordered_pairs": [[int(i), int(j)] for i, j in pairs],
        "thresholds": [float(x) for x in thresholds.detach().cpu().numpy()],
        "dataset_path": os.path.abspath(dataset_path),
    }
    summary_path = os.path.join(out_dir, f"{label}_summary.yaml")
    write_yaml(summary, summary_path)
    print(f"[DONE] Best model: {best_model_path}")
    print(f"[DONE] Summary: {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a multi-state next-hit committor model q_j(z).")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()
    raw = load_yaml(args.config)
    cfg = select_section(raw, "NEXT_HIT_COMMITTOR", "NEXT_HIT_TRAIN", "TRAIN")
    train_next_hit_committor(cfg)


if __name__ == "__main__":
    main()
