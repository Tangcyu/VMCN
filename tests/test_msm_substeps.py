from __future__ import annotations

import copy
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from tensorq.MSMlabel import pipeline


def _load_dispatcher():
    path = Path(__file__).resolve().parents[1] / "run.py"
    spec = importlib.util.spec_from_file_location("tensorq_run_dispatcher", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dispatcher = _load_dispatcher()


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("data", [("data", True)]),
        ("cluster", [("data", False), ("cluster", True)]),
        ("msm", [("data", False), ("cluster", False), ("msm", True)]),
        ("pcca", [("data", False), ("cluster", False), ("msm", False), ("pcca", True)]),
        (
            "core",
            [("data", False), ("cluster", False), ("msm", False), ("pcca", False), ("core", True)],
        ),
    ],
)
def test_substep_forces_only_target_and_reuses_upstream(monkeypatch, target, expected):
    calls = []

    def record(name, result):
        def fake(cfg, *args):
            calls.append((name, cfg["project"]["force"]))
            return result

        return fake

    monkeypatch.setattr(pipeline, "prepare_data", record("data", "table"))
    monkeypatch.setattr(pipeline, "cluster_microstates", record("cluster", "micro"))
    monkeypatch.setattr(pipeline, "build_msms", record("msm", "msms"))
    monkeypatch.setattr(pipeline, "analyze_macrostates", record("pcca", "pcca"))
    monkeypatch.setattr(pipeline, "build_core_label_datasets", record("core", None))
    monkeypatch.setattr(pipeline, "plot_implied_timescales", lambda *args: None)
    monkeypatch.setattr(pipeline, "plot_spectral_analysis", lambda *args: None)
    monkeypatch.setattr(pipeline, "plot_macrostates", lambda *args: None)
    monkeypatch.setattr(pipeline, "plot_ck", lambda *args: None)

    cfg = {"project": {"force": True}}
    original = copy.deepcopy(cfg)
    pipeline.run_pipeline(cfg, stage=target, reuse_upstream=True)

    assert calls == expected
    assert cfg == original


def test_unknown_msm_pipeline_stage_is_rejected():
    with pytest.raises(ValueError, match="Unknown MSM/core-label stage"):
        pipeline.run_pipeline({"project": {}}, stage="not-a-stage")


@pytest.mark.parametrize(
    ("requested", "canonical"),
    [
        ("data", "data"),
        ("microstates", "cluster"),
        ("msm", "msm"),
        ("pcca+", "pcca"),
        ("core-labeling", "core"),
        ("core-structures", "structures"),
    ],
)
def test_dispatcher_exposes_msm_substeps(monkeypatch, requested, canonical):
    calls = []
    monkeypatch.setattr(
        pipeline,
        "run_pipeline",
        lambda cfg, **kwargs: calls.append((cfg, kwargs)),
    )
    raw = {"MSM_CORE_LABEL": {"project": {"force": True}}}

    dispatcher.run_msm_core_label(raw, requested=requested)

    assert calls == [(raw["MSM_CORE_LABEL"], {"stage": canonical, "reuse_upstream": True})]


def test_dispatcher_rejects_unknown_msm_substep():
    with pytest.raises(ValueError, match="Unknown MSM/core-label substep"):
        dispatcher.run_msm_core_label({"MSM_CORE_LABEL": {}}, requested="bad")


def test_dispatcher_can_run_gradpath_clustering_directly(monkeypatch):
    calls = []
    fake_runner = types.ModuleType("tensorq.gradpath.runner")
    fake_runner.run_gradpath = lambda cfg: calls.append(cfg)
    monkeypatch.setitem(sys.modules, "tensorq.gradpath.runner", fake_runner)

    raw = {"GRADPATH": {"out_dir": "gradpath-test"}, "VORONOI_MERGE": {"enabled": False}}
    dispatcher.run_gradpath(raw, requested="clustering")

    assert calls == [raw["GRADPATH"]]
