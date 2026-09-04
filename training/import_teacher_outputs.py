#!/usr/bin/env python3
"""Validate provider-neutral teacher outputs into prompt/response SFT JSONL."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from prepare_dtf import clean_text


SENTENCE_END = re.compile(r"[.!?]+(?:\s|$)")
BOILERPLATE = ("как искусственный интеллект", "не могу ответить без контекста")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--prompts",
        type=Path,
        help="optional selected-prompts JSONL used when outputs contain only id/response",
    )
    parser.add_argument("--max-response-chars", type=int, default=350)
    parser.add_argument("--max-sentences", type=int, default=3)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    prompts_by_id: dict[str, str] = {}
    if args.prompts:
        with args.prompts.open("r", encoding="utf-8") as prompt_source:
            for line in prompt_source:
                item = json.loads(line)
                prompts_by_id[str(item["id"])] = str(item["prompt"])
    with args.input.open("r", encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line_number, line in enumerate(source, 1):
            stats["input_rows"] += 1
            item = json.loads(line)
            prompt_value = item.get("prompt") or prompts_by_id.get(str(item.get("id")))
            prompt = clean_text(prompt_value)
            response = clean_text(item.get("response"))
            if prompt is None or response is None:
                stats["removed_empty"] += 1
                continue
            folded = response.casefold()
            if len(response) > args.max_response_chars:
                stats["removed_too_long"] += 1
                continue
            if len(SENTENCE_END.findall(response)) > args.max_sentences:
                stats["removed_too_many_sentences"] += 1
                continue
            if any(marker in folded for marker in BOILERPLATE):
                stats["removed_boilerplate"] += 1
                continue
            identity = (prompt.casefold(), folded)
            if identity in seen:
                stats["removed_duplicate"] += 1
                continue
            seen.add(identity)
            target.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "response": response,
                        "source": "teacher-distillation",
                        "id": item.get("id", f"line-{line_number}"),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            stats["accepted"] += 1
    print(json.dumps(dict(sorted(stats.items())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
