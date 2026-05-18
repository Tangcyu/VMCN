#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorq.common.config import load_yaml, select_section
from tensorq.gradpath.runner import run_gradpath


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gradpaths from KDE/FEL-selected CV channel centers.")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()
    raw = load_yaml(args.config)
    cfg = dict(select_section(raw, "GRADPATH", "GradPath"))
    cfg.setdefault("selection_mode", "fel_kde")
    run_gradpath(cfg)


if __name__ == "__main__":
    main()
