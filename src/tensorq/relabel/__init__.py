"""State-label confidence diagnostics for committor-vector models.

Evaluates whether current state labels are reliable by analyzing:

- committor consistency (q vs assigned label)
- committor entropy/confidence
- q-argmax alternatives for low-consistency frames

Large disconnected kinetic groups inside the same high-confidence label core
are reported as basin-internal metastability signals.
"""

from .label_diagnostics import (
    DEFAULT_CONFIG,
    StateLabelDiagnostics,
    run_label_diagnostics,
    run_relabel as run_diagnostics,
)
from .relabel import propose_relabeling, run_relabel

__all__ = [
    "StateLabelDiagnostics",
    "run_label_diagnostics",
    "run_diagnostics",
    "run_relabel",
    "propose_relabeling",
    "DEFAULT_CONFIG",
]
