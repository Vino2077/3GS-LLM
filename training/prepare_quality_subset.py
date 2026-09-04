#!/usr/bin/env python3
"""Filter cleaned DTF pairs by child likes for controlled experiments."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cleaned", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-response-likes", type=int, required=True)
    args = parser.parse_args()
    if args.min_response_likes < 0:
        parser.error("--min-response-likes must be non-negative")
    args.output.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    for split in ("train", "validation"):
        source_path = args.cleaned / f"{split}.jsonl.gz"
        target_path = args.output / f"{split}.jsonl.gz"
        with gzip.open(source_path, "rt", encoding="utf-8") as source, gzip.open(
            target_path, "wt", encoding="utf-8", newline="\n"
        ) as target:
            for line in source:
                stats[f"{split}_input"] += 1
                record = json.loads(line)
                if int(record.get("response_likes", 0)) < args.min_response_likes:
                    continue
                target.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                stats[f"{split}_pairs"] += 1
    stats["min_response_likes"] = args.min_response_likes
    (args.output / "stats.json").write_text(
        json.dumps(dict(sorted(stats.items())), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(dict(sorted(stats.items())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
