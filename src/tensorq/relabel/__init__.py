"""State-label confidence diagnostics for committor-vector models.

Evaluates whether current state labels are reliable by analyzing:

- committor consistency (q vs assigned label)
- committor entropy/confidence
- q-argmax alternatives for low-consistency frames

Split/merge/missing-state decisions are intentionally not automated here.
Use the relabel hint table for triage, then validate candidate label changes
with targeted structural/CV inspection and retraining.
"""

from .label_diagnostics import DEFAULT_CONFIG, StateLabelDiagnostics, run_label_diagnostics, run_relabel

__all__ = ["StateLabelDiagnostics", "run_label_diagnostics", "run_relabel", "DEFAULT_CONFIG"]
