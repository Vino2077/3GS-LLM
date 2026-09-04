#!/usr/bin/env python3
"""Compare compiled-tokenizer BPE behavior with the tokenizers reference."""

from __future__ import annotations

import argparse
import gzip
import json
import struct
from pathlib import Path
from typing import Iterable

import numpy as np
from tokenizers import Tokenizer

from export_tokenizer import ALIGNMENT, HEADER, byte_to_unicode


class CompiledTokenizer:
    def __init__(self, path: Path) -> None:
        contents = path.read_bytes()
        unpacked = HEADER.unpack_from(contents)
        if unpacked[0] != b"3GSTOK1\0":
            raise ValueError("bad tokenizer magic")
        _, vocab, merge_count, *_special, decoded_length = unpacked[1:11]
        body = memoryview(contents)[ALIGNMENT:]
        self.base = np.frombuffer(body[: 256 * 2], dtype="<u2")
        offset_start = 256 * 2
        offset_end = offset_start + (vocab + 1) * 4
        self.offsets = np.frombuffer(body[offset_start:offset_end], dtype="<u4")
        self.decoded = body[offset_end : offset_end + decoded_length]
        merge_start = offset_end + decoded_length
        merge_start += merge_start % 2
        rows = np.frombuffer(
            body[merge_start : merge_start + merge_count * 6], dtype="<u2"
        ).reshape(-1, 3)
        self.merges = {
            (int(left), int(right)): (rank, int(result))
            for rank, (left, right, result) in enumerate(rows)
        }

    def encode_piece(self, data: bytes) -> list[int]:
        tokens = [int(self.base[value]) for value in data]
        while len(tokens) > 1:
            best: tuple[int, int, int] | None = None
            for index, pair in enumerate(zip(tokens, tokens[1:])):
                merge = self.merges.get(pair)
                if merge is not None and (best is None or merge[0] < best[0]):
                    best = (merge[0], index, merge[1])
            if best is None:
                break
            _, index, result = best
            tokens[index : index + 2] = [result]
        return tokens

    def decode(self, tokens: Iterable[int]) -> bytes:
        result = bytearray()
        for token in tokens:
            start = int(self.offsets[token])
            end = int(self.offsets[token + 1])
            result.extend(self.decoded[start:end])
        return bytes(result)


def corpus_texts(path: Path, limit: int) -> Iterable[str]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for index, line in enumerate(source):
            if index >= limit:
                break
            record = json.loads(line)
            yield record["parent"]
            yield record["response"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer_json", type=Path)
    parser.add_argument("tokenizer_bin", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--pairs", type=int, default=1000)
    args = parser.parse_args()

    reference = Tokenizer.from_file(str(args.tokenizer_json))
    compiled = CompiledTokenizer(args.tokenizer_bin)
    reverse = {character: value for value, character in byte_to_unicode().items()}
    checked = 0
    for text in corpus_texts(args.corpus, args.pairs):
        normalized = reference.normalizer.normalize_str(text)
        actual: list[int] = []
        for piece, _offsets in reference.pre_tokenizer.pre_tokenize_str(normalized):
            piece_bytes = bytes(reverse[character] for character in piece)
            actual.extend(compiled.encode_piece(piece_bytes))
        expected = reference.encode(text, add_special_tokens=False).ids
        if actual != expected:
            raise AssertionError(
                f"encoding mismatch for {text!r}: {actual} != {expected}"
            )
        if compiled.decode(actual).decode("utf-8") != normalized:
            raise AssertionError(f"decoding mismatch for {text!r}")
        checked += 1
    print(f"compiled tokenizer matches reference for {checked} real DTF texts")


if __name__ == "__main__":
    main()
