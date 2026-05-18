_EXPORTS = {
    "ChannelSelection": ("selection", "ChannelSelection"),
    "FelKdeSelectionResult": ("fel_selection", "FelKdeSelectionResult"),
    "GradientPath": ("shooting", "GradientPath"),
    "PathCluster": ("cluster", "PathCluster"),
    "build_channel_paths": ("shooting", "build_channel_paths"),
    "cluster_paths": ("cluster", "cluster_paths"),
    "cluster_paths_with_linkage": ("cluster", "cluster_paths_with_linkage"),
    "find_state_pairs": ("plot_runner", "find_state_pairs"),
    "find_transitions_above_threshold": ("state_p", "find_transitions_above_threshold"),
    "load_p_jump": ("state_p", "load_p_jump"),
    "pairwise_rmsd_matrix": ("cluster", "pairwise_rmsd_matrix"),
    "parse_state_endpoints": ("state_p", "parse_state_endpoints"),
    "run_gradpath": ("runner", "run_gradpath"),
    "run_gradpath_for_state_pairs": ("state_p", "run_gradpath_for_state_pairs"),
    "run_gradpath_plot": ("plot_runner", "run_gradpath_plot"),
    "select_channel_points": ("selection", "select_channel_points"),
    "select_fel_kde_centers": ("fel_selection", "select_fel_kde_centers"),
    "shoot_batch_to_state": ("shooting", "shoot_batch_to_state"),
    "shoot_to_state": ("shooting", "shoot_to_state"),
    "smooth_path": ("shooting", "smooth_path"),
    "weighted_center_path": ("cluster", "weighted_center_path"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = __import__(f"{__name__}.{module_name}", fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
