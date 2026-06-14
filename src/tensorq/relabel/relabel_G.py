from __future__ import annotations

import argparse
import os

import numpy as np

from ..common.config import ensure_dir, load_yaml, select_section, write_yaml
from . import relabel as _base_relabel

_BASE_PROPOSE_RELABELING = _base_relabel.propose_relabeling
_BASE_RUN_RELABEL = _base_relabel.run_relabel


def _gini_confidence(q_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return q confidence with normalized Gini impurity as the uncertainty score.

    This intentionally matches the tuple contract of `_entropy_confidence`:
    `(q_max, q_argmax, raw_uncertainty, normalized_uncertainty)`.
    The existing relabel pipeline can then be reused unchanged, with
    `analysis.entropy_cutoff` and `analysis.lagged_entropy_cutoff` interpreted
    as Gini cutoffs for this test entry point.
    """
    q = np.clip(np.asarray(q_values, dtype=np.float64), 0.0, 1.0)
    q_sum = np.sum(q, axis=1, keepdims=True)
    q_sum[q_sum <= 0.0] = 1.0
    q = q / q_sum
    q_max = np.max(q, axis=1)
    q_argmax = np.argmax(q, axis=1).astype(np.int64)
    gini = 1.0 - np.sum(q * q, axis=1)
    n_states = int(q.shape[1])
    denom = 1.0 - (1.0 / n_states) if n_states > 1 else 1.0
    gini_norm = gini / denom
    return q_max, q_argmax, gini, gini_norm


def propose_relabeling(*args, **kwargs):
    old = _base_relabel._entropy_confidence
    _base_relabel._entropy_confidence = _gini_confidence
    try:
        proposal = _BASE_PROPOSE_RELABELING(*args, **kwargs)
    finally:
        _base_relabel._entropy_confidence = old
    proposal["diagnostics"]["uncertainty_score"] = "normalized_gini"
    proposal["diagnostics"]["cutoff_note"] = (
        "analysis.entropy_cutoff and analysis.lagged_entropy_cutoff were "
        "interpreted as normalized Gini cutoffs in relabel_G.py."
    )
    return proposal


def run_relabel(*args, **kwargs):
    old_propose = _base_relabel.propose_relabeling
    _base_relabel.propose_relabeling = propose_relabeling
    try:
        summary = _BASE_RUN_RELABEL(*args, **kwargs)
    finally:
        _base_relabel.propose_relabeling = old_propose

    config = kwargs.get("config")
    if config is None and len(args) >= 3:
        config = args[2]
    config = {} if config is None else config
    summary["uncertainty_score"] = "normalized_gini"
    summary["notes"] = [
        "This test run used normalized Gini impurity G(x) instead of entropy H(x).",
        "The same config keys were reused; entropy_cutoff and lagged_entropy_cutoff are Gini cutoffs here.",
        *summary.get("notes", []),
    ]
    output_dir = ensure_dir(config.get("output_dir", "relabel_out"))
    summary_path = os.path.join(output_dir, "relabel_summary.yaml")
    if os.path.isdir(output_dir):
        write_yaml(summary, summary_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Apply relabeling with normalized Gini uncertainty.")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()

    raw = load_yaml(args.config)
    cfg = select_section(raw, "RELABEL", "TENSORQ_RELABEL")
    dataset_path = cfg.get("dataset", cfg.get("dataset_path"))
    if dataset_path is None:
        raise KeyError("Relabel config needs 'dataset' or 'dataset_path'.")
    model_path = cfg.get("model")
    if model_path is None:
        raise KeyError("Relabel config needs 'model' (path to trained checkpoint).")

    cfg["output_dir"] = ensure_dir(cfg.get("output_dir", cfg.get("out_dir", "relabel")))
    run_relabel(
        dataset_path=dataset_path,
        model_path=model_path,
        config=cfg,
        device=str(cfg.get("device", "cuda:0")),
        batch_size=int(cfg.get("batch_size", 65536)),
        dataset_stride=int(cfg.get("dataset_stride", 1)),
    )


if __name__ == "__main__":
    main()
