from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np

from .config import ensure_dir, stable_hash


def exists(path: str, force: bool = False) -> bool:
    return (not force) and os.path.exists(path)


def save_npz(path: str, *, manifest: Dict[str, Any] | None = None, **arrays: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    np.savez_compressed(path, **arrays)
    if manifest is not None:
        save_manifest(path + ".json", manifest)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=True))


def save_manifest(path: str, manifest: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    payload = dict(manifest)
    payload.setdefault("manifest_hash", stable_hash(manifest))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
