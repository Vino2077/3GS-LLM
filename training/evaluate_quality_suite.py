#!/usr/bin/env python3
"""Run the same human-readable prompt evaluation for several checkpoints."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("prompts", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--presets", nargs="+", default=["C"])
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    evaluator = Path(__file__).with_name("evaluate_quality.py")
    args.output.mkdir(parents=True, exist_ok=True)
    for checkpoint in args.checkpoints:
        stage = checkpoint.parent.name
        output = args.output / f"{stage}-{checkpoint.stem}.txt"
        subprocess.run(
            [
                sys.executable,
                str(evaluator),
                str(checkpoint),
                str(args.tokenizer),
                str(args.prompts),
                str(output),
                "--presets",
                *args.presets,
                "--seed",
                str(args.seed),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--device",
                args.device,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
