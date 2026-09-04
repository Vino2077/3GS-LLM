#!/usr/bin/env python3
"""Compile tokenizer.json into a small byte-oriented iPhone container."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


MAGIC = b"3GSTOK1\0"
VERSION = 1
HEADER = struct.Struct("<8s10I32s")
ALIGNMENT = 128
SPECIAL_NAMES = ("pad", "bos", "eos", "user", "assistant", "unk")


def byte_to_unicode() -> dict[int, str]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    code_points = byte_values[:]
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            code_points.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, map(chr, code_points)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    definition = json.loads(args.tokenizer.read_text(encoding="utf-8"))
    model = definition["model"]
    vocabulary: dict[str, int] = model["vocab"]
    vocab_size = len(vocabulary)
    by_id = [""] * vocab_size
    for token, identifier in vocabulary.items():
        by_id[identifier] = token
    if any(token == "" for token in by_id):
        raise ValueError("vocabulary IDs are not contiguous")

    special = {
        name: vocabulary[f"<{name}>"]
        for name in SPECIAL_NAMES
    }
    mapping = byte_to_unicode()
    reverse_mapping = {character: value for value, character in mapping.items()}
    base_ids = np.empty(256, dtype="<u2")
    for value, character in mapping.items():
        base_ids[value] = vocabulary[character]

    offsets = [0]
    decoded = bytearray()
    special_ids = set(special.values())
    for identifier, token in enumerate(by_id):
        if identifier not in special_ids:
            try:
                decoded.extend(reverse_mapping[character] for character in token)
            except KeyError as error:
                raise ValueError(
                    f"token {identifier} contains a non-byte-level character"
                ) from error
        offsets.append(len(decoded))

    merge_rows = []
    for rank, pair in enumerate(model["merges"]):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"unexpected merge representation at rank {rank}")
        left, right = pair
        result = left + right
        merge_rows.append((vocabulary[left], vocabulary[right], vocabulary[result]))
    merges = np.asarray(merge_rows, dtype="<u2")

    body = bytearray()
    body.extend(base_ids.tobytes())
    body.extend(np.asarray(offsets, dtype="<u4").tobytes())
    body.extend(decoded)
    if len(body) % 2:
        body.append(0)
    body.extend(merges.tobytes())
    body_hash = hashlib.sha256(body).digest()
    header = HEADER.pack(
        MAGIC,
        VERSION,
        vocab_size,
        len(merge_rows),
        *(special[name] for name in SPECIAL_NAMES),
        len(decoded),
        body_hash,
    )
    container = header + b"\0" * (ALIGNMENT - len(header)) + body

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(container)
    temporary.replace(args.output)
    manifest = {
        "format": "3gs-tokenizer-v1",
        "source_sha256": sha256_file(args.tokenizer),
        "container_sha256": sha256_file(args.output),
        "payload_sha256": body_hash.hex(),
        "container_bytes": len(container),
        "vocab_size": vocab_size,
        "merge_count": len(merge_rows),
        "decoded_bytes": len(decoded),
        "special_tokens": special,
        "pre_tokenizer": definition["pre_tokenizer"],
        "normalizer": definition["normalizer"],
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"exported tokenizer: {vocab_size} tokens, {len(merge_rows)} merges, "
        f"{len(container) / 1024:.1f} KiB; sha256={manifest['container_sha256']}"
    )


if __name__ == "__main__":
    main()
