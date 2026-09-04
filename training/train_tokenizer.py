#!/usr/bin/env python3
"""Train a byte-complete BPE tokenizer that can be ported to the iOS runtime."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<user>", "<assistant>", "<unk>"]


def text_iterator(path: Path) -> Iterable[str]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            yield f"<user>{record['parent']}<assistant>{record['response']}"


def count_lines(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return sum(1 for _ in source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path, help="clean train.jsonl.gz")
    parser.add_argument("output", type=Path)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        text_iterator(args.corpus), trainer=trainer, length=count_lines(args.corpus)
    )

    for expected_id, token in enumerate(SPECIAL_TOKENS):
        actual_id = tokenizer.token_to_id(token)
        if actual_id != expected_id:
            raise RuntimeError(f"{token} has id {actual_id}, expected {expected_id}")
    if tokenizer.get_vocab_size() != args.vocab_size:
        raise RuntimeError(
            f"tokenizer has {tokenizer.get_vocab_size()} entries, expected {args.vocab_size}"
        )

    tokenizer_path = args.output / "tokenizer.json"
    tokenizer.save(str(tokenizer_path), pretty=True)
    metadata = {
        "format": "3gs-byte-bpe-v1",
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": {
            token.strip("<>"): tokenizer.token_to_id(token) for token in SPECIAL_TOKENS
        },
        "byte_level_add_prefix_space": False,
        "byte_level_use_regex": True,
    }
    (args.output / "tokenizer_meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    probes = ["Привет, DTF!", "iPhone 3GS жив.", "Ну и зачем это нужно?"]
    for probe in probes:
        encoded = tokenizer.encode(probe)
        decoded = tokenizer.decode(encoded.ids)
        if decoded != probe:
            raise RuntimeError(f"tokenizer round-trip failed: {probe!r} -> {decoded!r}")
        print(f"{probe!r}: {len(encoded.ids)} tokens -> {encoded.ids}")


if __name__ == "__main__":
    main()
