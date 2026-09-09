#!/usr/bin/env python
"""Run the complete Approach-A ball synthetic validation and hard gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthetic.validate_ball import DEFAULT_SEED, format_summary_table, run_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.example.yaml",
        help="unified pipeline YAML (default: config.example.yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="artifact directory (default: <config directory>/out/synthetic/ball)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run every gate/sweep with shorter clips; omit for the release gate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"deterministic renderer seed (default: {DEFAULT_SEED})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print("Ball synthetic validation assumptions:")
    print("  camera: fixed to chassis; renderer K is exact")
    print("  segmentation: renderer-matched two-colour HSV")
    print("  Stages B/C: known circle and known ball-to-camera frame")
    print("  Stage E: automatic circle and pure-roll/pure-swivel frame calibration")
    print(f"  mode: {'quick (short clips)' if args.quick else 'full release gate'}")
    try:
        report = run_validation(
            args.config,
            output_dir=args.output,
            quick=args.quick,
            seed=args.seed,
            progress=print,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print()
    print(format_summary_table(report))
    return 0 if report.get("overall_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())

