from __future__ import annotations


def _as_lag_list(value, fallback=None):
    if value is None:
        value = fallback
    if value is None:
        return [1, 2, 5, 10, 20]
    if isinstance(value, (int, float)):
        value = [value]
    return [int(lag) for lag in value if int(lag) > 0]


def analysis_settings(config):
    """Compact shared settings for diagnose and relabel decisions.

    New configs should prefer the top-level ``analysis`` section. Legacy
    sections remain fallbacks so older YAML files keep running.
    """

    analysis = config.get("analysis", {})
    confidence = config.get("confidence", {})
    kinetics = config.get("kinetics", {})
    uncertainty = config.get("uncertainty", {})
    basin = config.get("basin_kinetic_groups", {})
    relabel = config.get("relabel", config.get("radical", {}))

    q_cutoff = float(analysis.get("q_cutoff", confidence.get("q_label_cutoff", 0.7)))
    entropy_cutoff = float(
        analysis.get("entropy_cutoff", confidence.get("entropy_cutoff_ambiguous", 0.5))
    )
    core_cutoff = float(
        analysis.get(
            "core_cutoff",
            basin.get("q_core_cutoff", confidence.get("confidence_cutoff_high", 0.8)),
        )
    )
    lag_list = _as_lag_list(analysis.get("lag_list", None), kinetics.get("lag_list", None))
    min_count = int(analysis.get("min_count", kinetics.get("min_valid_pairs", 50)))
    persistent_fraction = float(
        analysis.get(
            "persistent_fraction",
            relabel.get("candidate_min_persistent_fraction", 0.5),
        )
    )
    eigengap = float(analysis.get("eigengap", basin.get("min_eigengap", 0.05)))
    max_groups = int(analysis.get("max_groups", basin.get("max_macro_groups", 6)))
    min_group_size = int(analysis.get("min_group_size", basin.get("min_group_size", min_count)))
    random_seed = int(analysis.get("random_seed", relabel.get("random_seed", basin.get("random_seed", 0))))

    return {
        "q_cutoff": q_cutoff,
        "entropy_cutoff": entropy_cutoff,
        "core_cutoff": core_cutoff,
        "lag_list": lag_list,
        "min_count": min_count,
        "persistent_fraction": persistent_fraction,
        "eigengap": eigengap,
        "max_groups": max_groups,
        "min_group_size": min_group_size,
        "random_seed": random_seed,
        "lagged_entropy_cutoff": float(
            analysis.get("lagged_entropy_cutoff", uncertainty.get("lagged_entropy_cutoff", entropy_cutoff))
        ),
        "lagged_qmax_cutoff": float(
            analysis.get("lagged_qmax_cutoff", relabel.get("candidate_lagged_qmax_cutoff", q_cutoff))
        ),
        "min_slow_eigenvalue": float(
            analysis.get("min_slow_eigenvalue", basin.get("min_slow_eigenvalue", 0.8))
        ),
    }
