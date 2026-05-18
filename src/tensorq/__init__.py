"""TensorQ: shared tools for next-hit and pair-wise committors."""

__all__ = ["NextHitCommittorNet", "PairwiseCommittorNet"]


def __getattr__(name):
    if name == "NextHitCommittorNet":
        from .next_hit.model import NextHitCommittorNet

        return NextHitCommittorNet
    if name == "PairwiseCommittorNet":
        from .pairwise.model import PairwiseCommittorNet

        return PairwiseCommittorNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
