from __future__ import annotations

import os
import time
from copy import deepcopy

import numpy as np

from .diagnostics_io import save_results
from .lag_pair_utils import (
    build_lag_pairs,
    summarize_label_kinetics,
)

DEFAULT_CONFIG = {
    "output_dir": "diagnostics",
    "diagnostics": {
        "compute_kinetics": False,
        "profile_timing": False,
    },
    "output": {
        "save_per_frame_csv": False,
        "save_per_frame_npz": False,
        "save_q_values": True,
    },
    "confidence": {
        "entropy_cutoff_ambiguous": 0.5,
        "confidence_cutoff_high": 0.8,
        "q_label_cutoff": 0.7,
        "state_fraction_low_cutoff": 0.2,
        "state_mean_q_cutoff": 0.75,
        "reassign_dominance_cutoff": 0.65,
    },
    "kinetics": {
        "lag_list": [1, 2, 5, 10, 20],
        "min_valid_pairs": 50,
        "retention_cutoff": 0.6,
        "q_autocorr_cutoff": 0.5,
    },
}


def _deep_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


class StateLabelDiagnostics:
    def __init__(
        self,
        features,
        q_values,
        trajectory_index,
        frame_index,
        state_labels,
        core_state_labels,
        weights=None,
        config=None,
    ):
        self.features = np.asarray(features, dtype=np.float64)
        self.q_values = np.asarray(q_values, dtype=np.float64)
        self.trajectory_index = np.asarray(trajectory_index, dtype=np.int64)
        self.frame_index = np.asarray(frame_index, dtype=np.int64)
        self.state_labels = np.asarray(state_labels, dtype=np.int64)
        self.core_state_labels = np.asarray(core_state_labels, dtype=np.int64)
        self.weights = (
            np.asarray(weights, dtype=np.float64) if weights is not None else None
        )
        self.config = self._init_config(config)

        self.n_frames = int(self.features.shape[0])
        self.n_states = int(self.q_values.shape[1])

        self.per_frame = {}
        self.lag_pairs = {}
        self.diagnostic_features = None
        self.state_stats = []
        self.global_cluster_labels = None
        self.global_cluster_stats = []
        self.relabel_hints = []
        self.split_candidates = []
        self.merge_candidates = []
        self.missing_state_candidates = []

    def _init_config(self, config):
        cfg = deepcopy(DEFAULT_CONFIG)
        if config:
            _deep_update(cfg, config)
        return cfg

    # ------------------------------------------------------------------
    # Per-frame confidence metrics
    # ------------------------------------------------------------------

    def compute_committor_confidence(self):
        eps = 1e-12
        q = np.clip(self.q_values, eps, 1.0)
        q_max = np.max(q, axis=1)
        q_argmax = np.argmax(q, axis=1).astype(np.int64)
        q_entropy = -np.sum(q * np.log(q), axis=1)
        q_entropy_norm = q_entropy / np.log(q.shape[1])
        committor_confidence = 1.0 - q_entropy_norm

        self.per_frame["q_max"] = q_max
        self.per_frame["q_argmax"] = q_argmax
        self.per_frame["q_entropy"] = q_entropy
        self.per_frame["q_entropy_norm"] = q_entropy_norm
        self.per_frame["committor_confidence"] = committor_confidence

    def compute_label_consistency(self):
        consistency = np.full(self.n_frames, np.nan, dtype=np.float64)
        valid = self.state_labels >= 0
        idx = np.arange(self.n_frames)[valid]
        consistency[valid] = self.q_values[idx, self.state_labels[valid]]
        self.per_frame["label_consistency"] = consistency

    def compute_core_label_consistency(self):
        consistency = np.full(self.n_frames, np.nan, dtype=np.float64)
        valid = self.core_state_labels >= 0
        idx = np.arange(self.n_frames)[valid]
        consistency[valid] = self.q_values[idx, self.core_state_labels[valid]]
        self.per_frame["core_label_consistency"] = consistency

    # ------------------------------------------------------------------
    # State-level statistics
    # ------------------------------------------------------------------

    def compute_state_level_statistics(self):
        lag_list = self.config["kinetics"]["lag_list"]
        q_label_cutoff = self.config["confidence"]["q_label_cutoff"]
        min_valid = self.config["kinetics"]["min_valid_pairs"]
        compute_kinetics = bool(self.config.get("diagnostics", {}).get("compute_kinetics", True))
        kinetics = None
        if compute_kinetics:
            kinetics = summarize_label_kinetics(
                self.state_labels,
                self.q_values,
                self.lag_pairs,
                lag_list=lag_list,
                weights=self.weights,
                min_valid_pairs=min_valid,
                n_labels=self.n_states,
            )
        rows = []

        for i in range(self.n_states):
            mask = self.state_labels == i
            n_state = int(np.sum(mask))

            if n_state == 0:
                continue

            own_q = self.q_values[mask, i]
            entropy = self.per_frame["q_entropy_norm"][mask]

            row = {
                "state": i,
                "n_frames": n_state,
                "n_core_frames": int(np.sum(self.core_state_labels == i)),
                "mean_q_own": float(np.mean(own_q)),
                "median_q_own": float(np.median(own_q)),
                "fraction_low_consistency": float(np.mean(own_q < q_label_cutoff)),
                "mean_entropy_norm": float(np.mean(entropy)),
                "median_entropy_norm": float(np.median(entropy)),
            }

            if kinetics is not None:
                for lag in lag_list:
                    ret = kinetics["retention"][lag][i]
                    row[f"p_stay_lag_{lag}"] = float(ret) if not np.isnan(ret) else np.nan

                    q_ac = kinetics["q_autocorr"][lag][i]
                    row[f"q_autocorr_lag_{lag}"] = float(q_ac) if not np.isnan(q_ac) else np.nan

            rows.append(row)

        self.state_stats = rows
        return rows

    # ------------------------------------------------------------------
    # Confidence-based relabel hints
    # ------------------------------------------------------------------

    def build_relabel_hints(self):
        cfg = self.config["confidence"]
        q_label_cutoff = float(cfg["q_label_cutoff"])
        frac_low_cutoff = float(cfg.get("state_fraction_low_cutoff", 0.2))
        mean_q_cutoff = float(cfg.get("state_mean_q_cutoff", 0.75))
        entropy_cutoff = float(cfg.get("entropy_cutoff_ambiguous", 0.5))
        dominance_cutoff = float(cfg.get("reassign_dominance_cutoff", 0.65))

        hints = []
        q_argmax = self.per_frame["q_argmax"]
        label_consistency = self.per_frame["label_consistency"]
        entropy_norm = self.per_frame["q_entropy_norm"]

        for row in self.state_stats:
            state = int(row["state"])
            mask = self.state_labels == state
            low_mask = mask & np.isfinite(label_consistency) & (label_consistency < q_label_cutoff)
            n_low = int(np.sum(low_mask))

            dominant_alt = -1
            dominant_alt_fraction = 0.0
            if n_low:
                alt = q_argmax[low_mask]
                alt = alt[alt != state]
                if alt.size:
                    counts = np.bincount(alt, minlength=self.n_states)
                    dominant_alt = int(np.argmax(counts))
                    dominant_alt_fraction = float(counts[dominant_alt] / n_low)

            unreliable = (
                row["fraction_low_consistency"] >= frac_low_cutoff
                or row["mean_q_own"] < mean_q_cutoff
            )
            ambiguous = row["mean_entropy_norm"] >= entropy_cutoff

            if unreliable and dominant_alt >= 0 and dominant_alt_fraction >= dominance_cutoff:
                issue = "possible_reassignment_or_merge"
                next_step = (
                    f"Inspect frames labeled {state} whose q-argmax is {dominant_alt}; "
                    "if they occupy the same basin, relabel/reassign before considering a merge."
                )
            elif unreliable and ambiguous:
                issue = "ambiguous_or_missing_coordinate"
                next_step = (
                    "Inspect high-entropy frames in CV/structure space; this may be boundary contamination, "
                    "a missing descriptor, or a genuinely unresolved region."
                )
            elif unreliable:
                issue = "possible_mislabel_or_broad_state"
                next_step = (
                    "Inspect low-consistency frames for this label and compare their q-argmax distribution "
                    "against the assigned state."
                )
            elif ambiguous:
                issue = "transition_like_or_low_confidence"
                next_step = "Treat this state cautiously; entropy is high even if q_own is not catastrophically low."
            else:
                issue = "no_confidence_issue"
                next_step = "No immediate relabel action suggested from confidence diagnostics."

            hints.append({
                "state": state,
                "n_frames": int(row["n_frames"]),
                "n_low_consistency": n_low,
                "mean_q_own": float(row["mean_q_own"]),
                "fraction_low_consistency": float(row["fraction_low_consistency"]),
                "mean_entropy_norm": float(row["mean_entropy_norm"]),
                "dominant_alternative_state": dominant_alt,
                "dominant_alternative_fraction": dominant_alt_fraction,
                "issue": issue,
                "suggested_next_step": next_step,
            })

        self.relabel_hints = hints
        return hints

    # ------------------------------------------------------------------
    # Legacy candidate hooks
    # ------------------------------------------------------------------

    def detect_split_candidates(self):
        self.split_candidates = []
        return []

    def detect_merge_candidates(self):
        self.merge_candidates = []
        return []

    def detect_missing_state_candidates(self):
        self.missing_state_candidates = []
        return []

    # ------------------------------------------------------------------
    # Main workflow
    # ------------------------------------------------------------------

    def run_all(self):
        profile_timing = bool(self.config.get("diagnostics", {}).get("profile_timing", False))
        timings = {}

        def mark(name, start):
            if profile_timing:
                timings[name] = time.perf_counter() - start

        start = time.perf_counter()
        self.compute_committor_confidence()
        self.compute_label_consistency()
        self.compute_core_label_consistency()
        mark("confidence_metrics", start)

        compute_kinetics = bool(self.config.get("diagnostics", {}).get("compute_kinetics", True))

        if compute_kinetics:
            start = time.perf_counter()
            self.lag_pairs = build_lag_pairs(
                self.trajectory_index,
                self.frame_index,
                self.config["kinetics"]["lag_list"],
            )
            mark("build_lag_pairs", start)

        start = time.perf_counter()
        self.compute_state_level_statistics()
        mark("state_level_statistics", start)

        start = time.perf_counter()
        relabel_hints = self.build_relabel_hints()
        mark("relabel_hints", start)

        start = time.perf_counter()
        summary = self._build_summary()
        per_frame_data = self._build_per_frame_array()
        mark("build_outputs", start)
        if profile_timing:
            summary["timings"] = timings

        output_dir = self.config["output_dir"]
        results = {
            "state_confidence": self.state_stats,
            "relabel_hints": relabel_hints,
            "split_candidates": [],
            "merge_candidates": [],
            "missing_state_candidates": [],
            "cluster_table": self.global_cluster_stats,
            "summary": summary,
        }
        start = time.perf_counter()
        save_results(results, output_dir, per_frame=per_frame_data)
        mark("write_results", start)

        results["_per_frame"] = per_frame_data
        if profile_timing:
            results["timings"] = timings
            print("[RELABEL] Stage timings:")
            for name, elapsed in timings.items():
                print(f"  {name}: {elapsed:.3f} s")
        return results

    # ------------------------------------------------------------------
    # Summary and per-frame helpers
    # ------------------------------------------------------------------

    def _build_summary(self):
        state_confidence = []
        for row in self.state_stats:
            state_confidence.append({
                "state": row["state"],
                "n_frames": row["n_frames"],
                "n_core_frames": row.get("n_core_frames", 0),
                "mean_q_own": row["mean_q_own"],
                "median_q_own": row["median_q_own"],
                "fraction_low_consistency": row["fraction_low_consistency"],
                "mean_entropy_norm": row["mean_entropy_norm"],
            })

        return {
            "n_frames": self.n_frames,
            "n_states": self.n_states,
            "n_split_candidates": len(self.split_candidates),
            "n_merge_candidates": len(self.merge_candidates),
            "n_missing_state_candidates": len(self.missing_state_candidates),
            "state_confidence": state_confidence,
            "relabel_hints": self.relabel_hints,
            "split_candidates": self.split_candidates,
            "merge_candidates": self.merge_candidates,
            "missing_state_candidates": self.missing_state_candidates,
            "notes": {
                "candidate_detection": (
                    "Split/merge/missing-state candidate detection is intentionally disabled. "
                    "Use relabel_hints as confidence-based triage, then validate proposed changes "
                    "with targeted structural/CV inspection and retraining."
                )
            },
            "config": self.config,
        }

    def _build_per_frame_array(self):
        columns = {
            "frame_global_index": np.arange(self.n_frames, dtype=np.int64),
            "trajectory_index": self.trajectory_index,
            "frame_index": self.frame_index,
            "state_label": self.state_labels,
            "core_state_label": self.core_state_labels,
        }

        for key in ["q_argmax", "q_max", "q_entropy", "q_entropy_norm",
                     "committor_confidence", "label_consistency", "core_label_consistency"]:
            if key in self.per_frame:
                columns[key] = self.per_frame[key]

        if self.global_cluster_labels is not None:
            columns["cluster_id"] = self.global_cluster_labels

        if self.weights is not None:
            columns["weight"] = self.weights

        return columns


# ===================================================================
# Top-level wrappers
# ===================================================================

def _compute_frame_index(traj_id):
    frame_index = np.zeros(len(traj_id), dtype=np.int64)
    for traj in np.unique(traj_id):
        mask = traj_id == traj
        frame_index[mask] = np.arange(np.sum(mask), dtype=np.int64)
    return frame_index


def run_label_diagnostics(
    model,
    features,
    trajectory_index,
    frame_index,
    state_labels,
    core_state_labels,
    weights=None,
    config=None,
    device=None,
    batch_size=65536,
):
    if hasattr(model, "forward"):
        import torch

        if device is None:
            device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        model.eval()
        model.to(device)

        n_frames = int(features.shape[0])
        bs = int(batch_size)
        q_chunks = []

        with torch.no_grad():
            for start in range(0, n_frames, bs):
                end = min(n_frames, start + bs)
                batch = torch.as_tensor(features[start:end], dtype=torch.float32, device=device)
                q_chunks.append(model(batch).detach().cpu().numpy())

        q_values = np.vstack(q_chunks).astype(np.float64)
    else:
        q_values = np.asarray(model, dtype=np.float64)

    diagnostics = StateLabelDiagnostics(
        features=features,
        q_values=q_values,
        trajectory_index=trajectory_index,
        frame_index=frame_index,
        state_labels=state_labels,
        core_state_labels=core_state_labels,
        weights=weights,
        config=config,
    )

    return diagnostics.run_all()


def run_relabel(
    dataset_path,
    model_path,
    config=None,
    device="cuda:0",
    batch_size=65536,
    dataset_stride=1,
):
    """High-level relabel pipeline using dataset.pt and a trained model checkpoint.

    Follows the same pattern as ``next_hit/infer.py``: loads the dataset, loads
    the trained committor model onto the requested device, infers q-values for
    all frames, and runs the full diagnostic pipeline.

    Parameters
    ----------
    dataset_path : str
        Path to the .pt or .npz dataset produced by TENSORQ_LABEL.
    model_path : str
        Path to a TorchScript model or a next_hit checkpoint (.pt).
    config : dict or None
        Optional configuration merged with :data:`DEFAULT_CONFIG`.
    device : str
        Torch device string (e.g. ``"cuda:0"``, ``"cpu"``).
    batch_size : int
        Inference batch size.
    dataset_stride : int
        Stride applied to the dataset before inference.

    Returns
    -------
    dict
        Results dict with keys: ``state_confidence``, ``relabel_hints``,
        ``summary``, and ``_per_frame`` (for plotting). Legacy candidate keys
        are present but intentionally empty.
    """
    import torch

    from ..common.config import setup_device
    from ..common.data import apply_stride, infer_n_states, load_dataset, select_model_inputs
    from ..next_hit.predict import infer_probabilities, load_committor_model

    if config is None:
        config = {}

    device_obj = setup_device(device)

    pack = apply_stride(load_dataset(dataset_path), int(dataset_stride))
    n_states = infer_n_states(pack, config.get("n_states", None))
    model_features, input_meta = select_model_inputs(pack, config)

    model = load_committor_model(model_path, device_obj)
    q_values = infer_probabilities(
        model, model_features.float(), device_obj, batch_size=int(batch_size)
    )

    if q_values.ndim != 2 or q_values.shape[1] != n_states:
        raise RuntimeError(f"Model returned q shape {q_values.shape}, expected (_, {n_states}).")

    traj_id = pack.traj_id.numpy().astype(np.int64) if pack.traj_id is not None else np.zeros(q_values.shape[0], dtype=np.int64)
    frame_index = _compute_frame_index(traj_id)
    state_labels = pack.state.numpy().astype(np.int64)
    core_state_labels = state_labels.copy()
    weights = pack.weights.numpy().astype(np.float64)
    features = model_features.numpy().astype(np.float64)

    output_dir = config.get("output_dir", "relabel")
    config.setdefault("output_dir", output_dir)

    diagnostics = StateLabelDiagnostics(
        features=features,
        q_values=q_values,
        trajectory_index=traj_id,
        frame_index=frame_index,
        state_labels=state_labels,
        core_state_labels=core_state_labels,
        weights=weights,
        config=config,
    )

    results = diagnostics.run_all()

    output_cfg = config.get("output", {})
    if bool(output_cfg.get("save_q_values", True)):
        q_path = os.path.join(output_dir, "Q.npy")
        np.save(q_path, q_values.astype(np.float32))
        results["Q_npy"] = os.path.abspath(q_path)
    results["q_values"] = q_values

    return results
