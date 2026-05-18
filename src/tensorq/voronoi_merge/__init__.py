_EXPORTS = {
    "VoronoiAssignment": ("core", "VoronoiAssignment"),
    "assign_voronoi_cells": ("core", "assign_voronoi_cells"),
    "cell_probabilities": ("core", "cell_probabilities"),
    "kl_divergence": ("core", "kl_divergence"),
    "minimum_image_delta": ("core", "minimum_image_delta"),
    "voronoi_assignment": ("core", "voronoi_assignment"),
    "run_voronoi_merge": ("runner", "run_voronoi_merge"),
    "run_iterative_pathway_expansion": ("iterative", "run_iterative_pathway_expansion"),
    "decompose_pathway_segments": ("iterative", "decompose_pathway_segments"),
    "build_pathway_network": ("iterative", "build_pathway_network"),
    "find_all_reactive_pathways": ("iterative", "find_all_reactive_pathways"),
    "plot_shared_segments": ("plot", "plot_shared_segments"),
    "plot_pathway_network": ("plot", "plot_pathway_network"),
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
