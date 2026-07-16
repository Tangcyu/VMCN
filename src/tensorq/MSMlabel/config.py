from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Need pyyaml. Install with: pip install pyyaml") from exc


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg.setdefault("project", {})
    cfg["project"].setdefault("out_dir", "msmcv_out")
    cfg["project"].setdefault("seed", 2026)
    cfg["project"].setdefault("force", False)
    cfg["project"].setdefault("device", "cuda")
    return cfg


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def stage_path(cfg: Dict[str, Any], *parts: str) -> str:
    return os.path.join(cfg["project"]["out_dir"], *parts)
