from __future__ import annotations

import copy
from typing import Dict

from .cluster import cluster_microstates
from .core_labeling import build_core_label_datasets
from .core_structures import export_core_structures_from_dataset
from .data import prepare_data
from .export import write_summary
from .msm import build_msms
from .pcca import analyze_macrostates
from .plotting import plot_ck, plot_implied_timescales, plot_macrostates, plot_spectral_analysis


def run_pipeline(cfg: Dict, stage: str = "all", *, reuse_upstream: bool = False) -> None:
    """Run through *stage*, optionally forcing only the selected stage.

    With ``reuse_upstream=True``, prerequisite stages use a copy of the
    configuration with ``project.force`` disabled.  Existing checkpoints are
    therefore loaded even when the caller sets ``force: true`` to rebuild the
    requested stage.  A missing prerequisite is still built normally.
    """
    if stage == "structures":
        export_core_structures_from_dataset(cfg)
        return

    valid_stages = {"data", "cluster", "msm", "pcca", "core", "all"}
    if stage not in valid_stages:
        choices = ", ".join(sorted(valid_stages))
        raise ValueError(f"Unknown MSM/core-label stage {stage!r}; choose one of: {choices}.")

    upstream_cfg = cfg
    if reuse_upstream and stage != "all":
        upstream_cfg = copy.deepcopy(cfg)
        upstream_cfg.setdefault("project", {})["force"] = False

    def stage_config(name: str) -> Dict:
        if not reuse_upstream or stage == "all" or stage == name:
            return cfg
        return upstream_cfg

    table = prepare_data(stage_config("data"))
    if stage == "data":
        return

    micro = cluster_microstates(stage_config("cluster"), table)
    if stage == "cluster":
        return

    msms = build_msms(stage_config("msm"), table, micro)
    plot_implied_timescales(cfg, msms)
    plot_spectral_analysis(cfg, msms)
    if stage == "msm":
        return

    pcca = analyze_macrostates(stage_config("pcca"), table, micro, msms)
    plot_macrostates(cfg, table, micro, pcca)
    plot_ck(cfg, pcca)
    if stage == "pcca":
        return

    build_core_label_datasets(stage_config("core"), table, micro, pcca)
    if stage == "core":
        return

    summary = write_summary(cfg, msms, pcca)
    print(f"[ok] summary: {summary}")
