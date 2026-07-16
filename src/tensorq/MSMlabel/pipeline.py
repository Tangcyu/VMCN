from __future__ import annotations

from typing import Dict

from .cluster import cluster_microstates
from .core_labeling import build_core_label_datasets
from .core_structures import export_core_structures_from_dataset
from .data import prepare_data
from .export import write_summary
from .msm import build_msms
from .pcca import analyze_macrostates
from .plotting import plot_ck, plot_implied_timescales, plot_macrostates, plot_spectral_analysis


def run_pipeline(cfg: Dict, stage: str = "all") -> None:
    if stage == "structures":
        export_core_structures_from_dataset(cfg)
        return

    table = prepare_data(cfg)
    if stage == "data":
        return

    micro = cluster_microstates(cfg, table)
    if stage == "cluster":
        return

    msms = build_msms(cfg, table, micro)
    plot_implied_timescales(cfg, msms)
    plot_spectral_analysis(cfg, msms)
    if stage == "msm":
        return

    pcca = analyze_macrostates(cfg, table, micro, msms)
    plot_macrostates(cfg, table, micro, pcca)
    plot_ck(cfg, pcca)
    if stage == "pcca":
        return

    build_core_label_datasets(cfg, table, micro, pcca)
    if stage == "core":
        return

    summary = write_summary(cfg, msms, pcca)
    print(f"[ok] summary: {summary}")
