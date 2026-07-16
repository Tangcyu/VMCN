from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import stage_path


def write_summary(cfg: Dict, msms: Dict[int, Dict[str, np.ndarray]], pcca: Dict[int, Dict[int, Dict[str, np.ndarray]]]) -> str:
    rows = []
    for lag, by_m in pcca.items():
        for m, result in by_m.items():
            P = result["macro_transition"]
            rows.append(
                {
                    "lag": lag,
                    "m": m,
                    "min_self_transition": float(np.min(np.diag(P))),
                    "mean_self_transition": float(np.mean(np.diag(P))),
                    "max_ck_rmsd": float(np.max(result["ck_rmsd"])),
                    "min_residence_time": float(np.min(result["residence_times"])),
                    "max_escape_time": float(np.max(result["escape_times"])),
                }
            )
    path = stage_path(cfg, "summary.csv")
    pd.DataFrame(rows).to_csv(path, index=False)

    its_rows = []
    for lag, result in msms.items():
        for i, value in enumerate(result["implied_timescales"], start=1):
            its_rows.append({"lag": lag, "timescale_index": i, "timescale_frames": float(value)})
    pd.DataFrame(its_rows).to_csv(stage_path(cfg, "implied_timescales.csv"), index=False)
    return path
