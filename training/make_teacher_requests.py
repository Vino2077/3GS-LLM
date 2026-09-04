#!/usr/bin/env python3
"""Convert selected prompts to a provider-neutral teacher request JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompts", type=Path)
    parser.add_argument("teacher_prompt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    instruction = args.teacher_prompt.read_text(encoding="utf-8").strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.prompts.open("r", encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line in source:
            item = json.loads(line)
            request = {
                "id": item["id"],
                "system": instruction,
                "prompt": item["prompt"],
            }
            target.write(
                json.dumps(request, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            count += 1
    print(f"wrote {count} teacher requests -> {args.output}")


if __name__ == "__main__":
    main()
