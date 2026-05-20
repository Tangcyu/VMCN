#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorq.common.config import load_yaml, select_section
from tensorq.gradpath.state_p import run_gradpath_for_state_pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build gradpaths for state pairs above a P_jump probability threshold."
    )
    parser.add_argument("--config", required=True, help="YAML config path (GRADPATH section)")
    args = parser.parse_args()

    raw = load_yaml(args.config)
    cfg = dict(select_section(raw, "GRADPATH", "GradPath"))

    run_gradpath_for_state_pairs(cfg)


if __name__ == "__main__":
    main()
