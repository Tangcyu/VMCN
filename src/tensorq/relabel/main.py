from __future__ import annotations

import argparse
import os

from ..common.config import ensure_dir, load_yaml, select_section, setup_device, write_yaml
from ..common.data import apply_stride, load_dataset, select_model_inputs
from ..next_hit.predict import infer_probabilities, load_committor_model
from .label_diagnostics import run_relabel
from .plot import run_plots


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

    if bool(cfg.get("make_plots", True)):
        pack = apply_stride(load_dataset(dataset_path), int(cfg.get("dataset_stride", 1)))
        q_values = results.get("q_values")
        if q_values is None:
            device_obj = setup_device(cfg.get("device", "cuda:0"))
            model_features, _ = select_model_inputs(pack, cfg)
            model = load_committor_model(model_path, device_obj)
            q_values = infer_probabilities(
                model, model_features.float(), device_obj, int(cfg.get("batch_size", 65536))
            )
        plot_cfg = dict(cfg)
        plot_cfg.setdefault("format", cfg.get("format", "png"))
        plot_cfg.setdefault("planes", cfg.get("planes", cfg.get("cv_plane", [])))
        run_plots(pack, q_values, results, plot_cfg, output_dir)

    summary_path = os.path.join(output_dir, "diagnose_summary.yaml")
    write_yaml({
        "dataset": os.path.abspath(str(dataset_path)),
        "model": os.path.abspath(str(model_path)),
        "output_dir": os.path.abspath(output_dir),
        "n_frames": results["summary"]["n_frames"],
        "n_states": results["summary"]["n_states"],
        "n_split_candidates": results["summary"]["n_split_candidates"],
        "n_merge_candidates": results["summary"]["n_merge_candidates"],
        "n_missing_state_candidates": results["summary"]["n_missing_state_candidates"],
        "Q_npy": results.get("Q_npy"),
    }, summary_path)
    print(f"[DIAGNOSE] Summary: {summary_path}")
    print(f"[DIAGNOSE] Done. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
