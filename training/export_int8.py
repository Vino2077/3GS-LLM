#!/usr/bin/env python3
"""Export a checkpoint to the compact row-wise INT8 iPhone container."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
from pathlib import Path

import numpy as np
import torch

from model import ModelConfig


MAGIC = b"3GSLLM1\0"
VERSION = 1
HEADER = struct.Struct("<8s8I32s")
RECORD = struct.Struct("<HBB4IQQ")
TYPE_Q8_ROWWISE = 1
TYPE_FLOAT32 = 2
ALIGNMENT = 64


def align(stream: io.BytesIO) -> None:
    padding = (-stream.tell()) % ALIGNMENT
    if padding:
        stream.write(b"\0" * padding)


def tensor_names(config: ModelConfig) -> list[str]:
    names = ["token_embedding.weight"]
    for layer in range(config.n_layers):
        prefix = f"layers.{layer}."
        names.extend(
            [
                prefix + "attention_norm.weight",
                prefix + "attention.q_proj.weight",
                prefix + "attention.k_proj.weight",
                prefix + "attention.v_proj.weight",
                prefix + "attention.o_proj.weight",
                prefix + "ffn_norm.weight",
                prefix + "feed_forward.gate_proj.weight",
                prefix + "feed_forward.up_proj.weight",
                prefix + "feed_forward.down_proj.weight",
            ]
        )
    names.append("final_norm.weight")
    return names


def write_record(
    stream: io.BytesIO,
    name: str,
    tensor: torch.Tensor,
) -> dict[str, object]:
    array = tensor.detach().float().cpu().numpy()
    dimensions = list(array.shape)
    padded_dimensions = dimensions + [0] * (4 - len(dimensions))
    encoded_name = name.encode("ascii")

    if array.ndim == 2:
        maxima = np.max(np.abs(array), axis=1)
        scales = np.maximum(maxima / 127.0, np.finfo(np.float32).tiny).astype("<f4")
        quantized = np.clip(
            np.rint(array / scales[:, None]), -127, 127
        ).astype("i1")
        data = quantized.tobytes(order="C")
        auxiliary = scales.tobytes(order="C")
        tensor_type = TYPE_Q8_ROWWISE
        restored = quantized.astype(np.float32) * scales[:, None]
        mean_squared_error = float(np.mean((array - restored) ** 2))
        maximum_error = float(np.max(np.abs(array - restored)))
    elif array.ndim == 1:
        data = array.astype("<f4", copy=False).tobytes(order="C")
        auxiliary = b""
        tensor_type = TYPE_FLOAT32
        mean_squared_error = 0.0
        maximum_error = 0.0
    else:
        raise ValueError(f"unsupported tensor rank for {name}: {array.ndim}")

    stream.write(
        RECORD.pack(
            len(encoded_name),
            tensor_type,
            array.ndim,
            *padded_dimensions,
            len(data),
            len(auxiliary),
        )
    )
    stream.write(encoded_name)
    align(stream)
    stream.write(data)
    stream.write(auxiliary)
    align(stream)
    return {
        "name": name,
        "type": "q8-rowwise" if tensor_type == TYPE_Q8_ROWWISE else "float32",
        "shape": dimensions,
        "data_bytes": len(data),
        "auxiliary_bytes": len(auxiliary),
        "mean_squared_error": mean_squared_error,
        "maximum_error": maximum_error,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    state = checkpoint["model"]
    if not torch.equal(state["token_embedding.weight"], state["lm_head.weight"]):
        raise ValueError("checkpoint no longer has tied embedding/output weights")

    names = tensor_names(config)
    expected = set(names) | {"lm_head.weight"}
    if set(state) != expected:
        missing = sorted(expected - set(state))
        extra = sorted(set(state) - expected)
        raise ValueError(f"unexpected state dict; missing={missing}, extra={extra}")

    payload = io.BytesIO()
    payload.write(b"\0" * HEADER.size)
    align(payload)
    records = [write_record(payload, name, state[name]) for name in names]
    body = payload.getvalue()[ALIGNMENT * 2 :]
    body_hash = hashlib.sha256(body).digest()
    header = HEADER.pack(
        MAGIC,
        VERSION,
        config.vocab_size,
        config.context_length,
        config.d_model,
        config.n_layers,
        config.n_heads,
        config.d_ff,
        len(records),
        body_hash,
    )
    payload.seek(0)
    payload.write(header)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(payload.getvalue())
    temporary.replace(args.output)
    manifest = {
        "format": "3gs-llm-int8-v1",
        "model": config.to_dict(),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "container_sha256": sha256_file(args.output),
        "payload_sha256": body_hash.hex(),
        "container_bytes": args.output.stat().st_size,
        "tensors": records,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"exported {len(records)} tensors, {args.output.stat().st_size / 1024**2:.2f} MiB; "
        f"sha256={manifest['container_sha256']}"
    )


if __name__ == "__main__":
    main()
