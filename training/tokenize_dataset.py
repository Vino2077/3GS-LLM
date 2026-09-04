#!/usr/bin/env python3
"""Pack cleaned conversations into little-endian uint16 token streams."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm


def records(path: Path) -> Iterable[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def trim_pair(
    parent: list[int],
    response: list[int],
    payload: int,
    minimum_response_tokens: int = 64,
) -> tuple[list[int], list[int]]:
    if len(parent) + len(response) <= payload:
        return parent, response
    reserved_response = min(len(response), minimum_response_tokens, payload)
    parent_budget = min(len(parent), payload - reserved_response)
    response_budget = min(len(response), payload - parent_budget)
    trimmed_parent = parent[-parent_budget:] if parent_budget else []
    return trimmed_parent, response[:response_budget]


def write_split(
    input_path: Path,
    output_path: Path,
    mask_path: Path,
    offsets_path: Path,
    tokenizer: Tokenizer,
    context: int,
) -> dict[str, int]:
    ids = {name: tokenizer.token_to_id(f"<{name}>") for name in ("bos", "eos", "user", "assistant")}
    if any(value is None for value in ids.values()):
        raise RuntimeError("tokenizer is missing required special tokens")

    count = 0
    token_count = 0
    truncated = 0
    buffer: list[int] = []
    mask_buffer: list[int] = []
    response_tokens = 0
    offsets = [0]
    with output_path.open("wb") as target, mask_path.open("wb") as mask_target:
        for record in tqdm(records(input_path), desc=input_path.stem, unit="pairs"):
            parent = tokenizer.encode(str(record["parent"]), add_special_tokens=False).ids
            response = tokenizer.encode(str(record["response"]), add_special_tokens=False).ids
            original = len(parent) + len(response)
            parent, response = trim_pair(parent, response, context - 4)
            truncated += int(len(parent) + len(response) != original)
            sample = [ids["bos"], ids["user"], *parent, ids["assistant"], *response, ids["eos"]]
            response_mask = [0] * (3 + len(parent)) + [1] * (len(response) + 1)
            if len(sample) != len(response_mask):
                raise AssertionError("token and response-mask lengths differ")
            buffer.extend(sample)
            mask_buffer.extend(response_mask)
            count += 1
            token_count += len(sample)
            response_tokens += sum(response_mask)
            offsets.append(token_count)
            if len(buffer) >= 1_000_000:
                np.asarray(buffer, dtype="<u2").tofile(target)
                np.asarray(mask_buffer, dtype="u1").tofile(mask_target)
                buffer.clear()
                mask_buffer.clear()
        if buffer:
            np.asarray(buffer, dtype="<u2").tofile(target)
            np.asarray(mask_buffer, dtype="u1").tofile(mask_target)

    np.asarray(offsets, dtype="<u8").tofile(offsets_path)

    return {
        "pairs": count,
        "tokens": token_count,
        "response_tokens": response_tokens,
        "truncated_pairs": truncated,
        "offset_count": len(offsets),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cleaned", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--context", type=int, default=256)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    metadata: dict[str, object] = {
        "dtype": "uint16-le",
        "context": args.context,
        "vocab_size": tokenizer.get_vocab_size(),
        "offset_dtype": "uint64-le",
        "special_tokens": {
            name: tokenizer.token_to_id(f"<{name}>")
            for name in ("pad", "bos", "eos", "user", "assistant", "unk")
        },
        "truncation": {
            "prompt_side": "left",
            "response_side": "right",
            "minimum_response_tokens": 64,
        },
        "splits": {},
    }
    for split, source_name in (("train", "train.jsonl.gz"), ("validation", "validation.jsonl.gz")):
        result = write_split(
            args.cleaned / source_name,
            args.output / f"{split}.bin",
            args.output / f"{split}_response_mask.bin",
            args.output / f"{split}_offsets.bin",
            tokenizer,
            args.context,
        )
        metadata["splits"][split] = result  # type: ignore[index]
    (args.output / "dataset_meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
