from __future__ import annotations

import os
import time
from copy import deepcopy

import numpy as np

from .diagnostics_io import save_results
from .kinetic_groups import analyze_basin_kinetic_groups
from .lag_pair_utils import build_lag_pairs
from .settings import analysis_settings


UNCERTAINTY_CATEGORY_CODES = {
    "stable": 0,
    "mislabeled_metastate": 1,
    "missed_metastate": 2,
    "transition_state": 3,
    "unresolved_uncertain": 4,
}
UNCERTAINTY_CATEGORY_NAMES = {
    value: key for key, value in UNCERTAINTY_CATEGORY_CODES.items()
}


def _nanmean_by_frame(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    counts = np.sum(finite, axis=0)
    sums = np.where(finite, values, 0.0).sum(axis=0)
    out = np.full(values.shape[1], np.nan, dtype=np.float64)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return out


DEFAULT_CONFIG = {
    "output_dir": "diagnostics",
    "analysis": {
        "lag_list": [1, 2, 5, 10, 20],
        "q_cutoff": 0.7,
        "entropy_cutoff": 0.5,
        "core_cutoff": 0.8,
        "min_count": 50,
        "persistent_fraction": 0.5,
        "eigengap": 0.05,
        "max_groups": 6,
        "min_group_size": 50,
        "random_seed": 0,
    },
    "diagnostics": {
        "compute_lagged_entropy": False,
        "compute_kinetics": False,
        "compute_basin_kinetic_groups": True,
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
    },
    "basin_kinetic_groups": {
        "enabled": True,
        "q_core_mode": "own_high",
        "q_core_cutoff": 0.8,
        "require_q_argmax": True,
        "min_group_size": 50,
        "min_group_weight": 0.0,
        "min_valid_pairs": 50,
        "max_core_frames": 200000,
        "random_seed": 0,
        "analysis_lag": None,
        "min_microstates": 8,
        "max_microstates": 100,
        "target_frames_per_microstate": 500,
        "min_frames_per_microstate": 20,
        "max_macro_groups": 6,
        "min_slow_eigenvalue": 0.8,
        "min_eigengap": 0.05,
        "strong_eigengap": 0.12,
        "lag_list": None,
    },
    "uncertainty": {
        "lagged_entropy_cutoff": 0.5,
        "entropy_relief_cutoff": 0.15,
        "top2_min_probability": 0.2,
        "top2_margin_cutoff": 0.2,
        "min_category_fraction": 0.05,
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
        self.basin_kinetic_state_stats = []
        self.basin_kinetic_groups = []
        self.basin_kinetic_group_labels = None

    def _compute_lagged_entropy_enabled(self):
        diagnostics_cfg = self.config.get("diagnostics", {})
        return bool(
            diagnostics_cfg.get(
                "compute_lagged_entropy",
                diagnostics_cfg.get("compute_kinetics", False),
            )
        )

    def _compute_basin_kinetic_groups_enabled(self):
        diagnostics_cfg = self.config.get("diagnostics", {})
        group_cfg = self.config.get("basin_kinetic_groups", {})
        return bool(
            diagnostics_cfg.get(
                "compute_basin_kinetic_groups",
                group_cfg.get("enabled", True),
            )
        )

    def _lag_list(self):
        lag_cfg = self.config.get("lagged_entropy", {})
        settings = analysis_settings(self.config)
        lag_list = lag_cfg.get("lag_list", settings["lag_list"])
        if isinstance(lag_list, (int, float)):
            lag_list = [lag_list]
        return [int(lag) for lag in lag_list if int(lag) > 0]

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
        entropy_norm_denominator = np.log(q.shape[1]) if q.shape[1] > 1 else 1.0
        q_entropy_norm = q_entropy / entropy_norm_denominator
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

    def compute_lagged_committor_entropy(self):
        if not self.lag_pairs:
            return

        lagged_entropy_fields = []
        lagged_qmax_fields = []
        current_entropy = self.per_frame["q_entropy_norm"]

        for lag in self._lag_list():
            if lag not in self.lag_pairs:
                continue
            idx_t, idx_tau = self.lag_pairs[lag]
            lagged_entropy = np.full(self.n_frames, np.nan, dtype=np.float64)
            lagged_delta = np.full(self.n_frames, np.nan, dtype=np.float64)
            lagged_qmax = np.full(self.n_frames, np.nan, dtype=np.float64)
            lagged_qargmax = np.full(self.n_frames, -1, dtype=np.int64)

            lagged_entropy[idx_t] = current_entropy[idx_tau]
            lagged_delta[idx_t] = current_entropy[idx_tau] - current_entropy[idx_t]
            lagged_qmax[idx_t] = self.per_frame["q_max"][idx_tau]
            lagged_qargmax[idx_t] = self.per_frame["q_argmax"][idx_tau]

            entropy_key = f"lagged_q_entropy_norm_lag_{lag}"
            qmax_key = f"lagged_q_max_lag_{lag}"
            self.per_frame[entropy_key] = lagged_entropy
            self.per_frame[f"lagged_q_entropy_delta_lag_{lag}"] = lagged_delta
            self.per_frame[qmax_key] = lagged_qmax
            self.per_frame[f"lagged_q_argmax_lag_{lag}"] = lagged_qargmax
            lagged_entropy_fields.append(lagged_entropy)
            lagged_qmax_fields.append(lagged_qmax)

        if lagged_entropy_fields:
            entropy_stack = np.vstack(lagged_entropy_fields)
            mean_lagged_entropy = _nanmean_by_frame(entropy_stack)
            self.per_frame["mean_lagged_q_entropy_norm"] = mean_lagged_entropy
            self.per_frame["mean_lagged_q_entropy_delta"] = (
                mean_lagged_entropy - current_entropy
            )

        if lagged_qmax_fields:
            qmax_stack = np.vstack(lagged_qmax_fields)
            self.per_frame["mean_lagged_q_max"] = _nanmean_by_frame(qmax_stack)

    def classify_uncertainty_regions(self):
        settings = analysis_settings(self.config)
        cfg = self.config["confidence"]
        uncertainty_cfg = self.config.get("uncertainty", {})
        q_label_cutoff = float(settings["q_cutoff"])
        entropy_cutoff = float(settings["entropy_cutoff"])
        lagged_entropy_cutoff = float(
            settings["lagged_entropy_cutoff"]
        )
        entropy_relief_cutoff = float(uncertainty_cfg.get("entropy_relief_cutoff", 0.15))
        top2_min_probability = float(uncertainty_cfg.get("top2_min_probability", 0.2))
        top2_margin_cutoff = float(uncertainty_cfg.get("top2_margin_cutoff", 0.2))

        q_max = self.per_frame["q_max"]
        q_argmax = self.per_frame["q_argmax"]
        q = self.q_values
        if self.n_states == 1:
            top2_idx = np.column_stack([q_argmax, np.full(self.n_frames, -1, dtype=np.int64)])
            top2_probs = np.column_stack([q_max, np.zeros(self.n_frames, dtype=np.float64)])
        else:
            top2_idx = np.argpartition(q, -2, axis=1)[:, -2:]
            top2_probs = np.take_along_axis(q, top2_idx, axis=1)
            order = np.argsort(-top2_probs, axis=1)
            top2_idx = np.take_along_axis(top2_idx, order, axis=1)
            top2_probs = np.take_along_axis(top2_probs, order, axis=1)

        entropy = self.per_frame["q_entropy_norm"]
        label_consistency = self.per_frame["label_consistency"]
        has_lagged_entropy = "mean_lagged_q_entropy_norm" in self.per_frame
        lagged_entropy = self.per_frame.get("mean_lagged_q_entropy_norm", entropy)
        lagged_qmax = self.per_frame.get("mean_lagged_q_max", q_max)

        valid_label = self.state_labels >= 0
        low_consistency = (
            valid_label
            & np.isfinite(label_consistency)
            & (label_consistency < q_label_cutoff)
        )
        high_entropy = entropy >= entropy_cutoff
        persistent_high_entropy = (
            has_lagged_entropy
            &
            np.isfinite(lagged_entropy)
            & (lagged_entropy >= lagged_entropy_cutoff)
        )
        entropy_relief = (
            has_lagged_entropy
            &
            np.isfinite(lagged_entropy)
            & ((entropy - lagged_entropy) >= entropy_relief_cutoff)
        )
        lagged_committed = (
            has_lagged_entropy
            & np.isfinite(lagged_qmax)
            & (lagged_qmax >= q_label_cutoff)
        )
        two_state_mixture = (
            (top2_probs[:, 1] >= top2_min_probability)
            & ((top2_probs[:, 0] - top2_probs[:, 1]) <= top2_margin_cutoff)
        )
        confident_other_label = (
            valid_label
            & low_consistency
            & (q_argmax != self.state_labels)
            & (q_max >= q_label_cutoff)
        )

        category = np.full(
            self.n_frames,
            UNCERTAINTY_CATEGORY_CODES["stable"],
            dtype=np.int64,
        )
        category[high_entropy | low_consistency] = UNCERTAINTY_CATEGORY_CODES[
            "unresolved_uncertain"
        ]

        missed_metastate = (
            high_entropy
            & persistent_high_entropy
            & (q_max < q_label_cutoff)
            & ~two_state_mixture
        )
        transition_state = (
            high_entropy
            & two_state_mixture
            & entropy_relief
            & lagged_committed
        )

        category[missed_metastate] = UNCERTAINTY_CATEGORY_CODES["missed_metastate"]
        category[transition_state] = UNCERTAINTY_CATEGORY_CODES["transition_state"]
        category[confident_other_label] = UNCERTAINTY_CATEGORY_CODES[
            "mislabeled_metastate"
        ]

        self.per_frame["uncertainty_category"] = category
        self.per_frame["top1_state"] = top2_idx[:, 0].astype(np.int64, copy=False)
        self.per_frame["top2_state"] = top2_idx[:, 1].astype(np.int64, copy=False)
        self.per_frame["top1_probability"] = top2_probs[:, 0]
        self.per_frame["top2_probability"] = top2_probs[:, 1]
        self.per_frame["top2_margin"] = top2_probs[:, 0] - top2_probs[:, 1]

    def compute_basin_kinetic_groups(self):
        if not self.lag_pairs:
            return [], []
        state_rows, group_rows, group_labels = analyze_basin_kinetic_groups(
            self.state_labels,
            self.q_values,
            self.lag_pairs,
            self.config,
            weights=self.weights,
            n_states=self.n_states,
            features=self.features,
        )
        self.basin_kinetic_state_stats = state_rows
        self.basin_kinetic_groups = group_rows
        self.basin_kinetic_group_labels = group_labels
        return state_rows, group_rows

    # ------------------------------------------------------------------
    # State-level statistics
    # ------------------------------------------------------------------

    def compute_state_level_statistics(self):
        lag_list = self._lag_list()
        settings = analysis_settings(self.config)
        q_label_cutoff = settings["q_cutoff"]
        rows = []
        category = self.per_frame.get("uncertainty_category")
        min_category_fraction = float(
            self.config.get("uncertainty", {}).get("min_category_fraction", 0.05)
        )
        min_valid_pairs = int(settings["min_count"])

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

            if self.lag_pairs:
                for lag in lag_list:
                    entropy_key = f"lagged_q_entropy_norm_lag_{lag}"
                    delta_key = f"lagged_q_entropy_delta_lag_{lag}"
                    if entropy_key not in self.per_frame:
                        continue
                    lagged_entropy = self.per_frame[entropy_key][mask]
                    finite = np.isfinite(lagged_entropy)
                    n_valid = int(np.sum(finite))
                    row[f"n_lagged_pairs_lag_{lag}"] = n_valid
                    if n_valid >= min_valid_pairs:
                        row[f"mean_lagged_entropy_lag_{lag}"] = float(
                            np.mean(lagged_entropy[finite])
                        )
                        row[f"fraction_lagged_high_entropy_lag_{lag}"] = float(
                            np.mean(
                                lagged_entropy[finite]
                                >= self.config.get("uncertainty", {}).get(
                                    "lagged_entropy_cutoff",
                                    self.config["confidence"].get(
                                        "entropy_cutoff_ambiguous", 0.5
                                    ),
                                )
                            )
                        )
                    else:
                        row[f"mean_lagged_entropy_lag_{lag}"] = np.nan
                        row[f"fraction_lagged_high_entropy_lag_{lag}"] = np.nan

                    if delta_key in self.per_frame:
                        delta = self.per_frame[delta_key][mask]
                        finite_delta = np.isfinite(delta)
                        row[f"mean_lagged_entropy_delta_lag_{lag}"] = (
                            float(np.mean(delta[finite_delta]))
                            if int(np.sum(finite_delta)) >= min_valid_pairs
                            else np.nan
                        )

            if category is not None:
                fractions = {}
                for name, code in UNCERTAINTY_CATEGORY_CODES.items():
                    n_category = int(np.sum(mask & (category == code)))
                    row[f"n_{name}"] = n_category
                    row[f"fraction_{name}"] = float(n_category / n_state)
                    if name != "stable":
                        fractions[name] = row[f"fraction_{name}"]
                primary = max(fractions, key=fractions.get) if fractions else "stable"
                if fractions and fractions[primary] >= min_category_fraction:
                    row["primary_uncertainty_type"] = primary
                else:
                    row["primary_uncertainty_type"] = "stable"

            rows.append(row)

        self.state_stats = rows
        return rows

    # ------------------------------------------------------------------
    # Confidence-based relabel hints
    # ------------------------------------------------------------------

    def build_relabel_hints(self):
        settings = analysis_settings(self.config)
        cfg = self.config["confidence"]
        q_label_cutoff = float(settings["q_cutoff"])
        frac_low_cutoff = float(cfg.get("state_fraction_low_cutoff", 0.2))
        mean_q_cutoff = float(cfg.get("state_mean_q_cutoff", 0.75))
        entropy_cutoff = float(settings["entropy_cutoff"])
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
            primary_uncertainty = row.get("primary_uncertainty_type", "stable")

            if primary_uncertainty == "mislabeled_metastate":
                issue = "mislabeled_metastate"
                next_step = (
                    "Frames in this label are confidently assigned to a different committor basin; "
                    "inspect/reassign those frames before adding new states."
                )
            elif primary_uncertainty == "missed_metastate":
                issue = "missed_metastate_or_missing_descriptor"
                next_step = (
                    "High entropy persists at the lagged endpoint, so this is less transition-like; "
                    "inspect whether these frames form a stable missing basin or require better features."
                )
            elif primary_uncertainty == "transition_state":
                issue = "transition_state_like"
                next_step = (
                    "Entropy is high at the frame but drops at the lagged endpoint, consistent with "
                    "a transition region between the top committor destinations."
                )
            elif unreliable and dominant_alt >= 0 and dominant_alt_fraction >= dominance_cutoff:
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
                "primary_uncertainty_type": primary_uncertainty,
                "fraction_mislabeled_metastate": float(row.get("fraction_mislabeled_metastate", 0.0)),
                "fraction_missed_metastate": float(row.get("fraction_missed_metastate", 0.0)),
                "fraction_transition_state": float(row.get("fraction_transition_state", 0.0)),
                "fraction_unresolved_uncertain": float(row.get("fraction_unresolved_uncertain", 0.0)),
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

        compute_lagged_entropy = self._compute_lagged_entropy_enabled()
        compute_basin_groups = self._compute_basin_kinetic_groups_enabled()

        if compute_lagged_entropy or compute_basin_groups:
            start = time.perf_counter()
            self.lag_pairs = build_lag_pairs(
                self.trajectory_index,
                self.frame_index,
                self._lag_list(),
            )
            mark("build_lag_pairs", start)

            if compute_lagged_entropy:
                start = time.perf_counter()
                self.compute_lagged_committor_entropy()
                mark("lagged_committor_entropy", start)

        if compute_basin_groups:
            start = time.perf_counter()
            self.compute_basin_kinetic_groups()
            mark("basin_kinetic_groups", start)

        start = time.perf_counter()
        self.classify_uncertainty_regions()
        mark("uncertainty_classification", start)

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
            "basin_kinetic_state_stats": self.basin_kinetic_state_stats,
            "basin_kinetic_groups": self.basin_kinetic_groups,
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
                "primary_uncertainty_type": row.get("primary_uncertainty_type", "stable"),
                "fraction_mislabeled_metastate": row.get("fraction_mislabeled_metastate", 0.0),
                "fraction_missed_metastate": row.get("fraction_missed_metastate", 0.0),
                "fraction_transition_state": row.get("fraction_transition_state", 0.0),
                "fraction_unresolved_uncertain": row.get("fraction_unresolved_uncertain", 0.0),
            })
            for key, value in row.items():
                if (
                    key.startswith("mean_lagged_entropy")
                    or key.startswith("fraction_lagged_high_entropy")
                    or key.startswith("n_lagged_pairs")
                ):
                    state_confidence[-1][key] = value

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
            "basin_kinetic_state_stats": self.basin_kinetic_state_stats,
            "basin_kinetic_groups": self.basin_kinetic_groups,
            "notes": {
                "uncertainty_category_codes": UNCERTAINTY_CATEGORY_CODES,
                "lagged_entropy": (
                    "Lagged entropy uses H_norm(q(x_{t+tau})) for each trajectory-safe lag pair. "
                    "Transition-state-like regions have high current entropy that relaxes to a "
                    "lower-entropy, committed lagged endpoint; missed-metastate-like regions keep "
                    "high entropy after the lag; mislabeled-metastate regions have low assigned-label "
                    "consistency but high confidence for another existing state."
                ),
                "candidate_detection": (
                    "Split/merge/missing-state candidate detection is intentionally disabled. "
                    "Use relabel_hints as confidence-based triage, then validate proposed changes "
                    "with targeted structural/CV inspection and retraining."
                ),
                "basin_kinetic_groups": (
                    "Kinetic groups are estimated by a local spectral MSM inside each high-confidence "
                    "label core: feature-space microstates, trajectory-safe lagged transitions, "
                    "eigenvalue/eigengap group-count selection, then slow-eigenvector macrostate "
                    "assignment. Multiple robust groups inside one label suggest a split."
                ),
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

        for key in [
            "q_argmax",
            "q_max",
            "q_entropy",
            "q_entropy_norm",
            "committor_confidence",
            "label_consistency",
            "core_label_consistency",
            "mean_lagged_q_entropy_norm",
            "mean_lagged_q_entropy_delta",
            "mean_lagged_q_max",
            "uncertainty_category",
            "top1_state",
            "top2_state",
            "top1_probability",
            "top2_probability",
            "top2_margin",
        ]:
            if key in self.per_frame:
                columns[key] = self.per_frame[key]

        for key in sorted(self.per_frame):
            if key.startswith("lagged_q_"):
                columns[key] = self.per_frame[key]

        if self.global_cluster_labels is not None:
            columns["cluster_id"] = self.global_cluster_labels

        if self.basin_kinetic_group_labels is not None:
            columns["basin_kinetic_group"] = self.basin_kinetic_group_labels

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
