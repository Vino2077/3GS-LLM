#!/usr/bin/env python3
"""Select deterministic, unique DTF parent comments for teacher generation."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path

from prepare_dtf import clean_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cleaned_train", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--min-prompt-chars", type=int, default=8)
    parser.add_argument("--max-prompt-chars", type=int, default=600)
    parser.add_argument("--min-response-likes", type=int, default=0)
    parser.add_argument("--allow-small", action="store_true")
    args = parser.parse_args()
    if not args.allow_small and not 30_000 <= args.count <= 100_000:
        parser.error("--count must be 30000-100000 (or use --allow-small for tests)")

    rng = random.Random(args.seed)
    reservoir: list[dict[str, object]] = []
    seen: set[bytes] = set()
    eligible = 0
    with gzip.open(args.cleaned_train, "rt", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if int(record.get("response_likes", 0)) < args.min_response_likes:
                continue
            prompt = clean_text(record.get("parent"))
            if prompt is None or not args.min_prompt_chars <= len(prompt) <= args.max_prompt_chars:
                continue
            identity = hashlib.blake2b(
                prompt.casefold().encode("utf-8"), digest_size=16
            ).digest()
            if identity in seen:
                continue
            seen.add(identity)
            eligible += 1
            candidate = {
                "id": "dtf-" + identity.hex(),
                "prompt": prompt,
                "source_parent_likes": int(record.get("parent_likes", 0)),
                "source_response_likes": int(record.get("response_likes", 0)),
            }
            if len(reservoir) < args.count:
                reservoir.append(candidate)
            else:
                replacement = rng.randrange(eligible)
                if replacement < args.count:
                    reservoir[replacement] = candidate

    if len(reservoir) < args.count:
        raise RuntimeError(
            f"only {len(reservoir)} eligible unique prompts, requested {args.count}"
        )
    reservoir.sort(key=lambda item: str(item["id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as target:
        for item in reservoir:
            target.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(
        f"selected {len(reservoir)} of {eligible} eligible unique prompts -> {args.output}"
    )


if __name__ == "__main__":
    main()
