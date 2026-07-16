#!/usr/bin/env python3
"""Run the staged MSM-to-committor-vector workflow.

The individual packages keep their native command-line interfaces.  This
module only supplies a single dispatcher and a single top-level pipeline
configuration.  Relative paths are interpreted from the directory in which
the command is launched, just like the existing scripts.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


STAGE_ALIASES = {
    "0": "msmcorelabel",
    "msm": "msmcorelabel",
    "msmcorelabel": "msmcorelabel",
    "msm-to-core": "msmcorelabel",
    "msm_to_core": "msmcorelabel",
    "1": "committorvector",
    "committor": "committorvector",
    "committorvector": "committorvector",
    "next-hit": "committorvector",
    "next_hit": "committorvector",
    "2": "gradpath",
    "gradpath": "gradpath",
    "grad_path": "gradpath",
    "3": "relabel",
    "relabel": "relabel",
    "relabeling": "relabel",
}

DEFAULT_STAGE_CONFIGS = {
    "msmcorelabel": "configs/0.MSMcorelabel.yaml",
    "committorvector": "configs/1.Committorvector.yaml",
    "gradpath": "configs/2.Gradpath.yaml",
    "relabel": "configs/3.Relabel.yaml",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return value


def canonical_stage(value: str) -> str:
    key = str(value).strip().lower()
    if key in {"all", "pipeline"}:
        return "all"
    try:
        return STAGE_ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(sorted(set(STAGE_ALIASES.values())))
        raise ValueError(f"Unknown --step {value!r}. Choose one of: {choices}, all") from exc


def _load_stage_config(main_path: Path, stage: str) -> tuple[dict[str, Any], Path]:
    """Load a stage config directly, or through the top-level config.yaml."""

    raw = load_yaml(main_path)
    pipeline = raw.get("pipeline", raw.get("PIPELINE"))
    is_dispatch_config = isinstance(pipeline, dict) and any(
        key in pipeline or f"{key}_config" in pipeline for key in DEFAULT_STAGE_CONFIGS
    )
    if is_dispatch_config:
        configured = pipeline.get(stage) or pipeline.get(f"{stage}_config")
        configured = configured or DEFAULT_STAGE_CONFIGS[stage]
        stage_path = Path(str(configured))
        if not stage_path.is_absolute():
            stage_path = (main_path.parent / stage_path).resolve()
        return load_yaml(stage_path), stage_path
    return raw, main_path


def _section(raw: dict[str, Any], *names: str) -> dict[str, Any]:
    for name in names:
        value = raw.get(name)
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError(f"Configuration section {name!r} must be a mapping.")
            return copy.deepcopy(value)
    return copy.deepcopy(raw)


def _configured_substeps(raw: dict[str, Any], default: Iterable[str]) -> list[str]:
    pipeline = raw.get("pipeline", raw.get("PIPELINE", {}))
    if isinstance(pipeline, dict) and pipeline.get("substeps") is not None:
        value = pipeline["substeps"]
    else:
        value = list(default)
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("pipeline.substeps must be a list or a string.")
    return [str(item).strip().lower().replace("-", "_") for item in value]


def run_msm_core_label(raw: dict[str, Any]) -> None:
    from tensorq.MSMlabel.pipeline import run_pipeline

    # This is the same complete pipeline selected by:
    # python -m tensorq.MSMlabel.cli all CONFIG
    config = _section(raw, "MSM_CORE_LABEL", "MSMCORELABEL")
    run_pipeline(config, stage="all")


def run_committor_vector(raw: dict[str, Any], requested: str | None = None) -> None:

    from tensorq.next_hit.infer import run_inference
    from tensorq.next_hit.plot import run as plot
    from tensorq.next_hit.rate_constant import run as rate_constant
    from tensorq.next_hit.train import train_next_hit_committor

    substeps = [requested] if requested else _configured_substeps(
        raw, ("train", "infer", "plot", "rate_constant")
    )
    actions = {
        "train": ("NEXT_HIT_COMMITTOR", "NEXT_HIT_TRAIN", "TRAIN", train_next_hit_committor),
        "infer": ("NEXT_HIT_INFER", "INFER", run_inference),
        "plot": ("NEXT_HIT_PLOT", "PLOT", plot),
        "rate": ("NEXT_HIT_RATE", "RATE_CONSTANT", rate_constant),
        "rate_constant": ("NEXT_HIT_RATE", "RATE_CONSTANT", rate_constant),
    }
    for name in substeps:
        normalized = name.replace("rateconstant", "rate_constant")
        if normalized not in actions:
            raise ValueError(
                f"Unknown committor-vector substep {name!r}; use train, infer, plot, "
                "or rate_constant."
            )
        *section_names, function = actions[normalized]
        config = _section(raw, *section_names)
        print(f"\n[PIPELINE] committorvector/{normalized}")
        function(config)


def run_gradpath(raw: dict[str, Any], requested: str | None = None) -> None:
    grad_cfg = _section(raw, "GRADPATH", "GradPath")
    merge_cfg = _section(raw, "VORONOI_MERGE", "GRADPATH_MERGY", "gradpath_mergy")
    substeps = [requested] if requested else _configured_substeps(
        raw, ("pathfinding", "clustering", "merging")
    )
    pathfinding_done = False
    for name in substeps:
        if name == "cluster":
            name = "clustering"
        if name in {"pathfinding", "shooting", "paths"}:
            automatic = bool(grad_cfg.get("automatic_pairs", False)) or any(
                key in grad_cfg for key in ("p_jump", "p_jump_path")
            )
            print(f"\n[PIPELINE] gradpath/{name}")
            from tensorq.gradpath.runner import run_gradpath as run_single_gradpath

            if automatic:
                from tensorq.gradpath.state_p import run_gradpath_for_state_pairs

                run_gradpath_for_state_pairs(grad_cfg)
            else:
                run_single_gradpath(grad_cfg)
            # The native gradpath runner performs weighted path clustering in
            # the same call as gradient shooting.
            pathfinding_done = True
        elif name == "clustering":
            if pathfinding_done:
                print("[PIPELINE] gradpath/clustering already completed by pathfinding")
            else:
                print("\n[PIPELINE] gradpath/clustering (native runner bundles shooting + clustering)")
                run_single_gradpath(grad_cfg)
                pathfinding_done = True
        elif name in {"plot", "plots"}:
            from tensorq.gradpath.plot_runner import run_gradpath_plot

            print("\n[PIPELINE] gradpath/plot")
            if bool(grad_cfg.get("automatic_pairs", False)) or any(
                key in grad_cfg for key in ("p_jump", "p_jump_path")
            ):
                # Plotting's --auto behaviour is represented by looping over
                # the state-pair directories through its public function.
                from tensorq.gradpath.plot_runner import find_state_pairs

                out_dir = str(grad_cfg.get("out_dir", "./gradpath"))
                for state_i, state_j in find_state_pairs(out_dir):
                    pair_cfg = dict(grad_cfg, state_i=state_i, state_j=state_j)
                    run_gradpath_plot(pair_cfg)
            else:
                run_gradpath_plot(grad_cfg)
        elif name in {"merging", "merge", "voronoi_merge", "voronoi"}:
            if merge_cfg.get("enabled", True) is False:
                print("[PIPELINE] gradpath/merging disabled (set VORONOI_MERGE.enabled: true)")
                continue
            from tensorq.voronoi_merge.runner import run_voronoi_merge

            print("\n[PIPELINE] gradpath/merging")
            run_voronoi_merge(merge_cfg)
        else:
            raise ValueError(
                f"Unknown gradpath substep {name!r}; use pathfinding, clustering, "
                "plot, or merging."
            )


def _method_config(raw: dict[str, Any], method: str) -> dict[str, Any]:
    config = _section(raw, "RELABEL", "TENSORQ_RELABEL")
    override = raw.get(method, {})
    if not isinstance(override, dict):
        override = config.get(method, {})
    if isinstance(override, dict):
        for key, value in override.items():
            if key not in {"enabled", "output_dir", "output_dataset"}:
                config[key] = copy.deepcopy(value)
        if override.get("output_dir") is not None:
            config["output_dir"] = override["output_dir"]
        if override.get("output_dataset") is not None:
            relabel = dict(config.get("relabel", {}))
            relabel["output_dataset"] = override["output_dataset"]
            config["relabel"] = relabel
    return config


def run_relabel(raw: dict[str, Any], requested: str | None = None) -> None:
    from tensorq.relabel import relabel as entropy_module
    from tensorq.relabel import relabel_G as gini_module
    from tensorq.relabel.label_diagnostics import run_relabel as diagnose

    base = _section(raw, "RELABEL", "TENSORQ_RELABEL")
    substeps = [requested] if requested else _configured_substeps(
        raw, ("diagnose", "relabel_entropy", "relabel_gini")
    )
    dataset = base.get("dataset", base.get("dataset_path"))
    model = base.get("model")
    if dataset is None or model is None:
        raise KeyError("RELABEL requires dataset/dataset_path and model.")

    for name in substeps:
        name = name.replace("diagnosing", "diagnose").replace("diagnostics", "diagnose")
        if name in {"diagnose", "diagnose_entropy", "entropy_diagnose"}:
            config = _method_config(raw, "entropy")
            config["output_dir"] = base.get("diagnostics_output_dir", config.get("output_dir", "./relabel/diagnostics"))
            print("\n[PIPELINE] relabel/diagnose (H = normalized committor entropy)")
            diagnose(
                dataset_path=dataset,
                model_path=model,
                config=config,
                device=str(config.get("device", "cuda:0")),
                batch_size=int(config.get("batch_size", 65536)),
                dataset_stride=int(config.get("dataset_stride", 1)),
            )
        elif name in {"relabel", "relabel_entropy", "entropy", "h"}:
            config = _method_config(raw, "entropy")
            print("\n[PIPELINE] relabel/relabel_entropy (H = normalized committor entropy)")
            entropy_module.run_relabel(
                dataset_path=dataset,
                model_path=model,
                config=config,
                device=str(config.get("device", "cuda:0")),
                batch_size=int(config.get("batch_size", 65536)),
                dataset_stride=int(config.get("dataset_stride", 1)),
            )
        elif name in {"relabel_gini", "gini", "g"}:
            config = _method_config(raw, "gini")
            print("\n[PIPELINE] relabel/relabel_gini (G = normalized Gini impurity)")
            gini_module.run_relabel(
                dataset_path=dataset,
                model_path=model,
                config=config,
                device=str(config.get("device", "cuda:0")),
                batch_size=int(config.get("batch_size", 65536)),
                dataset_stride=int(config.get("dataset_stride", 1)),
            )
        else:
            raise ValueError(
                f"Unknown relabel substep {name!r}; use diagnose, relabel_entropy, or relabel_gini."
            )


def run_stage(stage: str, main_config: Path, substep: str | None = None) -> None:
    raw, stage_path = _load_stage_config(main_config, stage)
    print(f"[PIPELINE] config: {stage_path}")
    if stage == "msmcorelabel":
        if substep is not None:
            raise ValueError("MSM core labeling has no run.py substeps; use --step msmcorelabel.")
        run_msm_core_label(raw)
    elif stage == "committorvector":
        run_committor_vector(raw, substep)
    elif stage == "gradpath":
        run_gradpath(raw, substep)
    elif stage == "relabel":
        run_relabel(raw, substep)
    else:  # pragma: no cover - canonical_stage prevents this
        raise ValueError(f"Unsupported stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TensorQ MSM-to-committor-vector workflow.")
    parser.add_argument(
        "--step",
        required=True,
        help="Stage: 0/msmcorelabel, 1/committorvector, 2/gradpath, 3/relabel, or all.",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config.yaml"),
        help="Top-level config.yaml or a direct stage YAML (default: ./config.yaml).",
    )
    parser.add_argument(
        "--substep",
        default=None,
        help="Optional single substep, e.g. train, rate_constant, pathfinding, merging, or gini.",
    )
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    stage = canonical_stage(args.step)
    if stage == "all":
        if args.substep is not None:
            raise ValueError("--substep cannot be combined with --step all.")
        for name in ("msmcorelabel", "committorvector", "gradpath", "relabel"):
            run_stage(name, config_path)
    else:
        run_stage(stage, config_path, args.substep)


if __name__ == "__main__":
    main()
