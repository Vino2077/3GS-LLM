#!/usr/bin/env python3
"""Report sizes and simple characteristics of DTF child-like subsets."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cleaned", type=Path)
    parser.add_argument("--thresholds", type=int, nargs="+", default=[10, 25, 50])
    args = parser.parse_args()
    thresholds = sorted(set(args.thresholds))
    report: dict[str, object] = {"thresholds": {}}
    totals = {threshold: 0 for threshold in thresholds}
    char_totals = {threshold: 0 for threshold in thresholds}
    for split in ("train", "validation"):
        split_counts = {threshold: 0 for threshold in thresholds}
        with gzip.open(
            args.cleaned / f"{split}.jsonl.gz", "rt", encoding="utf-8"
        ) as source:
            for line in source:
                record = json.loads(line)
                likes = int(record.get("response_likes", 0))
                for threshold in thresholds:
                    if likes >= threshold:
                        split_counts[threshold] += 1
                        totals[threshold] += 1
                        char_totals[threshold] += len(record["response"])
        for threshold in thresholds:
            entry = report["thresholds"].setdefault(  # type: ignore[union-attr]
                str(threshold), {}
            )
            entry[f"{split}_pairs"] = split_counts[threshold]
    for threshold in thresholds:
        entry = report["thresholds"][str(threshold)]  # type: ignore[index]
        entry["total_pairs"] = totals[threshold]
        entry["average_response_chars"] = (
            char_totals[threshold] / totals[threshold] if totals[threshold] else 0.0
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
