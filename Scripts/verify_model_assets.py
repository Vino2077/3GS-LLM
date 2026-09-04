#!/usr/bin/env python3
"""Validate the binary model and tokenizer containers before IPA packaging."""

from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path


MODEL_HEADER = struct.Struct("<8s8I32s")
MODEL_RECORD = struct.Struct("<HBB4IQQ")
TOKENIZER_HEADER = struct.Struct("<8s10I32s")
ALIGNMENT = 128
RECORD_ALIGNMENT = 64


def aligned(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def expected_tensors(layers: int, width: int, feed_forward: int, vocab: int):
    yield "token_embedding.weight", 1, (vocab, width)
    for layer in range(layers):
        prefix = f"layers.{layer}."
        yield prefix + "attention_norm.weight", 2, (width,)
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            yield prefix + f"attention.{projection}.weight", 1, (width, width)
        yield prefix + "ffn_norm.weight", 2, (width,)
        yield prefix + "feed_forward.gate_proj.weight", 1, (feed_forward, width)
        yield prefix + "feed_forward.up_proj.weight", 1, (feed_forward, width)
        yield prefix + "feed_forward.down_proj.weight", 1, (width, feed_forward)
    yield "final_norm.weight", 2, (width,)


def verify_model(path: Path) -> None:
    contents = path.read_bytes()
    if len(contents) < ALIGNMENT:
        raise ValueError("model container is truncated")
    unpacked = MODEL_HEADER.unpack_from(contents)
    magic = unpacked[0]
    version, vocab, context, width, layers, heads, feed_forward, count = unpacked[1:9]
    payload_hash = unpacked[9]
    if magic != b"3GSLLM1\0" or version != 1:
        raise ValueError("unsupported model container")
    if (vocab, context, width, layers, heads, feed_forward) != (
        8192,
        256,
        384,
        8,
        6,
        1024,
    ):
        raise ValueError("model dimensions do not match the iPhone runtime")
    if hashlib.sha256(contents[ALIGNMENT:]).digest() != payload_hash:
        raise ValueError("model payload SHA-256 mismatch")

    expected = list(expected_tensors(layers, width, feed_forward, vocab))
    if count != len(expected):
        raise ValueError(f"expected {len(expected)} tensors, found {count}")
    offset = ALIGNMENT
    for expected_name, expected_type, expected_shape in expected:
        fields = MODEL_RECORD.unpack_from(contents, offset)
        name_length, tensor_type, dimensions = fields[:3]
        shape = tuple(fields[3 : 3 + dimensions])
        data_bytes, auxiliary_bytes = fields[7:9]
        offset += MODEL_RECORD.size
        name = contents[offset : offset + name_length].decode("ascii")
        offset = aligned(offset + name_length, RECORD_ALIGNMENT)
        if (name, tensor_type, shape) != (
            expected_name,
            expected_type,
            expected_shape,
        ):
            raise ValueError(
                f"unexpected tensor {(name, tensor_type, shape)}; "
                f"expected {(expected_name, expected_type, expected_shape)}"
            )
        elements = 1
        for dimension in shape:
            elements *= dimension
        expected_data = elements if tensor_type == 1 else elements * 4
        expected_auxiliary = shape[0] * 4 if tensor_type == 1 else 0
        if (data_bytes, auxiliary_bytes) != (expected_data, expected_auxiliary):
            raise ValueError(f"invalid byte lengths for {name}")
        offset = aligned(
            offset + data_bytes + auxiliary_bytes,
            RECORD_ALIGNMENT,
        )
    if offset != len(contents):
        raise ValueError("model container has trailing or missing bytes")
    print(f"model OK: {count} tensors, {len(contents) / 1024**2:.2f} MiB")


def verify_tokenizer(path: Path) -> None:
    contents = path.read_bytes()
    if len(contents) < ALIGNMENT:
        raise ValueError("tokenizer container is truncated")
    unpacked = TOKENIZER_HEADER.unpack_from(contents)
    magic = unpacked[0]
    values = unpacked[1:11]
    payload_hash = unpacked[11]
    version, vocab, merge_count, *special, decoded_bytes = values
    if magic != b"3GSTOK1\0" or version != 1:
        raise ValueError("unsupported tokenizer container")
    if vocab != 8192 or merge_count != vocab - 262:
        raise ValueError("unexpected tokenizer dimensions")
    if special != [0, 1, 2, 3, 4, 5]:
        raise ValueError("unexpected special-token IDs")
    body = contents[ALIGNMENT:]
    if hashlib.sha256(body).digest() != payload_hash:
        raise ValueError("tokenizer payload SHA-256 mismatch")

    base_bytes = 256 * 2
    offset_bytes = (vocab + 1) * 4
    base_ids = struct.unpack_from("<256H", body)
    if any(identifier >= vocab for identifier in base_ids):
        raise ValueError("base byte token ID outside vocabulary")
    offsets = struct.unpack_from(f"<{vocab + 1}I", body, base_bytes)
    if offsets[0] != 0 or offsets[-1] != decoded_bytes:
        raise ValueError("invalid decoder offsets")
    if any(left > right for left, right in zip(offsets, offsets[1:])):
        raise ValueError("decoder offsets are not monotonic")
    merge_offset = aligned(base_bytes + offset_bytes + decoded_bytes, 2)
    expected_bytes = merge_offset + merge_count * 3 * 2
    if expected_bytes != len(body):
        raise ValueError("tokenizer container has trailing or missing bytes")
    print(
        f"tokenizer OK: {vocab} tokens, {merge_count} merges, "
        f"{len(contents) / 1024:.1f} KiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("tokenizer", type=Path)
    args = parser.parse_args()
    verify_model(args.model)
    verify_tokenizer(args.tokenizer)


if __name__ == "__main__":
    main()
