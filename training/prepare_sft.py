#!/usr/bin/env python3
"""Validate prompt/response JSONL and create leakage-safe SFT splits."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, TextIO

from prepare_dtf import clean_text


SENTENCE_END = re.compile(r"[.!?]+(?:\s|$)")


def open_text(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", newline="\n")
    return path.open(mode, encoding="utf-8", newline="\n")


def records(path: Path) -> Iterable[dict[str, object]]:
    with open_text(path, "r") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"record is not an object at {path}:{line_number}")
            yield value


def validation_split(prompt: str, percent: int) -> bool:
    digest = hashlib.blake2b(prompt.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % 100 < percent


def load_excluded_prompts(path: Path | None) -> set[str]:
    if path is None:
        return set()
    values = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["prompt"]).casefold() for item in values}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="JSONL[.gz] with prompt/response")
    parser.add_argument("output", type=Path)
    parser.add_argument("--validation-percent", type=int, default=2)
    parser.add_argument("--max-prompt-chars", type=int, default=800)
    parser.add_argument("--max-response-chars", type=int, default=600)
    parser.add_argument("--max-sentences", type=int, default=3)
    parser.add_argument("--exclude-prompts", type=Path)
    args = parser.parse_args()
    if not 1 <= args.validation_percent <= 20:
        parser.error("--validation-percent must be between 1 and 20")

    args.output.mkdir(parents=True, exist_ok=True)
    excluded = load_excluded_prompts(args.exclude_prompts)
    seen: set[tuple[str, str]] = set()
    stats: Counter[str] = Counter()
    train_path = args.output / "train.jsonl.gz"
    validation_path = args.output / "validation.jsonl.gz"
    with open_text(train_path, "w") as train, open_text(validation_path, "w") as val:
        for record in records(args.input):
            stats["input_rows"] += 1
            prompt = clean_text(record.get("prompt"))
            response = clean_text(record.get("response"))
            if prompt is None or response is None:
                stats["removed_empty"] += 1
                continue
            if len(prompt) > args.max_prompt_chars or len(response) > args.max_response_chars:
                stats["removed_too_long"] += 1
                continue
            if prompt.casefold() in excluded:
                stats["removed_eval_leakage"] += 1
                continue
            if prompt.casefold() == response.casefold():
                stats["removed_echo"] += 1
                continue
            sentence_count = len(SENTENCE_END.findall(response))
            if sentence_count > args.max_sentences:
                stats["removed_too_many_sentences"] += 1
                continue
            identity = (prompt.casefold(), response.casefold())
            if identity in seen:
                stats["removed_duplicate"] += 1
                continue
            seen.add(identity)
            output_record = {
                "parent": prompt,
                "response": response,
                "source": str(record.get("source", "sft")),
            }
            target = val if validation_split(prompt, args.validation_percent) else train
            target.write(
                json.dumps(output_record, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            stats[
                "validation_pairs" if target is val else "train_pairs"
            ] += 1

    if not stats["train_pairs"] or not stats["validation_pairs"]:
        raise RuntimeError("SFT split is empty; provide more examples or change the split")
    (args.output / "stats.json").write_text(
        json.dumps(dict(sorted(stats.items())), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(dict(sorted(stats.items())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
