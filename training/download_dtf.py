#!/usr/bin/env python3
"""Download only the Parquet shards required for the DTF training corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REPOSITORY = "SubMaroon/DTF_Comments_Responses_Counts"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    result = snapshot_download(
        repo_id=REPOSITORY,
        repo_type="dataset",
        allow_patterns=["data/*.parquet", "README.md"],
        local_dir=args.output,
    )
    shards = sorted((Path(result) / "data").glob("train-*.parquet"))
    if len(shards) != 8:
        raise RuntimeError(f"expected 8 Parquet shards, found {len(shards)}")
    total = sum(path.stat().st_size for path in shards)
    print(f"downloaded {len(shards)} shards ({total / 1024**2:.1f} MiB) to {result}")


if __name__ == "__main__":
    main()
