from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MSM/PCCA+ workflow for committor-vector state selection."
    )
    parser.add_argument(
        "command",
        choices=["data", "cluster", "msm", "pcca", "core", "structures", "all"],
        help="Pipeline stage to run.",
    )
    parser.add_argument("config", help="YAML configuration file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_pipeline(cfg, stage=args.command)


if __name__ == "__main__":
    main()
