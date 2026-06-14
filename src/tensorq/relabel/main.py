from __future__ import annotations

import argparse
import os

from ..common.config import ensure_dir, load_yaml, select_section
from .label_diagnostics import run_relabel


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose state-label confidence for committor-vector models."
    )
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

    output_dir = ensure_dir(cfg.get("output_dir", cfg.get("out_dir", "relabel")))
    cfg["output_dir"] = output_dir

    results = run_relabel(
        dataset_path=dataset_path,
        model_path=model_path,
        config=cfg,
        device=str(cfg.get("device", "cuda:0")),
        batch_size=int(cfg.get("batch_size", 65536)),
        dataset_stride=int(cfg.get("dataset_stride", 1)),
    )

    summary_path = os.path.join(output_dir, "diagnostic_summary.yaml")
    print(f"[DIAGNOSE] Summary: {summary_path}")
    print(f"[DIAGNOSE] Frames: {results['summary']['n_frames']}")
    print(f"[DIAGNOSE] States: {results['summary']['n_states']}")
    print(f"[DIAGNOSE] Done. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
