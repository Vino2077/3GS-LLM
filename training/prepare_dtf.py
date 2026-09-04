#!/usr/bin/env python3
"""Clean DTF parent/reply pairs without loading the 3.8 GB table into RAM."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from tqdm import tqdm


COLUMNS = [
    "post_title",
    "parent_comment",
    "child_comment",
    "comment_id_parent",
    "comment_id_child",
    "parent_likes",
    "child_likes",
    "reply_count",
    "parent_comment_tox",
    "child_comment_tox",
]

DELETED_MARKERS = (
    "этот материал был удален по просьбе автора",
    "этот материал был удалн по просьбе автора",
    "комментарий недоступен",
    "комментарий удален автором поста",
)

WHITESPACE = re.compile(r"[ \t\v\f]+")
NEWLINES = re.compile(r"\n{3,}")


def clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = unicodedata.normalize("NFKC", html.unescape(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(WHITESPACE.sub(" ", line).strip() for line in text.split("\n"))
    text = NEWLINES.sub("\n\n", text).strip()
    folded = text.casefold().strip(" .!?;:")
    if not text or any(marker in folded for marker in DELETED_MARKERS):
        return None
    return text


def split_is_validation(post: str) -> bool:
    digest = hashlib.blake2b(post.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % 100 == 0


def iter_rows(paths: Iterable[Path], batch_size: int) -> Iterable[dict[str, Any]]:
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=COLUMNS):
            columns = batch.to_pydict()
            for index in range(batch.num_rows):
                yield {name: columns[name][index] for name in COLUMNS}


def open_jsonl(path: Path):
    return gzip.open(path, "wt", encoding="utf-8", newline="\n", compresslevel=6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="downloaded Hugging Face snapshot")
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-parent-chars", type=int, default=1600)
    parser.add_argument("--max-response-chars", type=int, default=1200)
    args = parser.parse_args()

    shards = sorted((args.input / "data").glob("train-*.parquet"))
    if len(shards) != 8:
        raise RuntimeError(f"expected 8 Parquet shards, found {len(shards)}")
    args.output.mkdir(parents=True, exist_ok=True)

    stats: Counter[str] = Counter()
    seen_children: set[int | tuple[str, str]] = set()
    train_path = args.output / "train.jsonl.gz"
    validation_path = args.output / "validation.jsonl.gz"

    total_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in shards)
    with open_jsonl(train_path) as train_file, open_jsonl(validation_path) as val_file:
        for row in tqdm(
            iter_rows(shards, args.batch_size), total=total_rows, unit="pairs"
        ):
            stats["input_rows"] += 1
            parent = clean_text(row["parent_comment"])
            response = clean_text(row["child_comment"])
            if parent is None or response is None:
                stats["removed_deleted_or_empty"] += 1
                continue
            if len(parent) > args.max_parent_chars or len(response) > args.max_response_chars:
                stats["removed_too_long"] += 1
                continue
            if parent.casefold() == response.casefold():
                stats["removed_echo"] += 1
                continue

            child_id = row["comment_id_child"]
            identity: int | tuple[str, str]
            if isinstance(child_id, int) and child_id > 0:
                identity = child_id
            else:
                identity = (parent, response)
            if identity in seen_children:
                stats["removed_duplicate_child"] += 1
                continue
            seen_children.add(identity)

            post = clean_text(row["post_title"]) or ""
            record = {
                "parent": parent,
                "response": response,
                "parent_likes": int(row["parent_likes"] or 0),
                "response_likes": int(row["child_likes"] or 0),
                "reply_count": int(row["reply_count"] or 0),
                "parent_toxicity": float(row["parent_comment_tox"] or 0.0),
                "response_toxicity": float(row["child_comment_tox"] or 0.0),
            }
            output = val_file if split_is_validation(post) else train_file
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            output.write("\n")
            stats["validation_pairs" if output is val_file else "train_pairs"] += 1
            if record["response_toxicity"] >= 0.8:
                stats["retained_high_toxicity"] += 1

    stats["unique_child_ids"] = len(seen_children)
    stats_path = args.output / "stats.json"
    stats_path.write_text(
        json.dumps(dict(sorted(stats.items())), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(stats_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
