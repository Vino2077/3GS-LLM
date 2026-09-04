"""Exact Python reference for exported row-wise INT8 weights and activations."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ModelConfig, ThreeGSModel


HEADER = struct.Struct("<8s8I32s")
RECORD = struct.Struct("<HBB4IQQ")
ALIGNMENT = 64
TYPE_Q8_ROWWISE = 1
TYPE_FLOAT32 = 2


def aligned(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def quantize_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scales = (weight.float().abs().amax(dim=1) / 127.0).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    values = torch.round(weight.float() / scales[:, None]).clamp(-127, 127)
    return values.to(torch.int8), scales


class QuantizedEmbedding(nn.Module):
    def __init__(self, values: torch.Tensor, scales: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("values", values.to(torch.int8))
        self.register_buffer("scales", scales.float())

    @classmethod
    def from_weight(cls, weight: torch.Tensor) -> "QuantizedEmbedding":
        return cls(*quantize_rows(weight))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        values = F.embedding(tokens, self.values).float()
        scales = F.embedding(tokens, self.scales[:, None])
        return values * scales


class QuantizedLinear(nn.Module):
    def __init__(self, values: torch.Tensor, scales: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("values", values.to(torch.int8))
        self.register_buffer("scales", scales.float())

    @classmethod
    def from_weight(cls, weight: torch.Tensor) -> "QuantizedLinear":
        return cls(*quantize_rows(weight))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        float_inputs = inputs.float()
        input_scales = (
            float_inputs.abs().amax(dim=-1, keepdim=True) / 127.0
        ).clamp_min(torch.finfo(torch.float32).tiny)
        quantized = torch.round(float_inputs / input_scales).clamp(-127, 127)
        accumulators = F.linear(quantized, self.values.float())
        return accumulators * input_scales * self.scales


def replace_quantized_matrices(
    model: ThreeGSModel,
    matrices: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> None:
    def linear(name: str, weight: torch.Tensor) -> QuantizedLinear:
        if matrices is None:
            return QuantizedLinear.from_weight(weight)
        return QuantizedLinear(*matrices[name])

    embedding_weight = model.token_embedding.weight.detach()
    if matrices is None:
        embedding = QuantizedEmbedding.from_weight(embedding_weight)
    else:
        embedding = QuantizedEmbedding(*matrices["token_embedding.weight"])
    model.token_embedding = embedding
    model.lm_head = QuantizedLinear(embedding.values, embedding.scales)
    for index, layer in enumerate(model.layers):
        prefix = f"layers.{index}."
        layer.attention.q_proj = linear(
            prefix + "attention.q_proj.weight", layer.attention.q_proj.weight
        )
        layer.attention.k_proj = linear(
            prefix + "attention.k_proj.weight", layer.attention.k_proj.weight
        )
        layer.attention.v_proj = linear(
            prefix + "attention.v_proj.weight", layer.attention.v_proj.weight
        )
        layer.attention.o_proj = linear(
            prefix + "attention.o_proj.weight", layer.attention.o_proj.weight
        )
        layer.feed_forward.gate_proj = linear(
            prefix + "feed_forward.gate_proj.weight",
            layer.feed_forward.gate_proj.weight,
        )
        layer.feed_forward.up_proj = linear(
            prefix + "feed_forward.up_proj.weight",
            layer.feed_forward.up_proj.weight,
        )
        layer.feed_forward.down_proj = linear(
            prefix + "feed_forward.down_proj.weight",
            layer.feed_forward.down_proj.weight,
        )


def quantize_model(model: ThreeGSModel) -> ThreeGSModel:
    replace_quantized_matrices(model)
    return model


def parse_exported_model(
    path: Path,
) -> tuple[
    ModelConfig,
    dict[str, tuple[torch.Tensor, torch.Tensor]],
    dict[str, torch.Tensor],
]:
    contents = path.read_bytes()
    unpacked = HEADER.unpack_from(contents)
    magic, version = unpacked[:2]
    if magic != b"3GSLLM1\0" or version != 1:
        raise ValueError("unsupported model container")
    vocab, context, width, layers, heads, d_ff, tensor_count = unpacked[2:9]
    config = ModelConfig(
        vocab_size=vocab,
        context_length=context,
        d_model=width,
        n_layers=layers,
        n_heads=heads,
        d_ff=d_ff,
    )
    if hashlib.sha256(contents[128:]).digest() != unpacked[9]:
        raise ValueError("model payload hash mismatch")

    matrices: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    vectors: dict[str, torch.Tensor] = {}
    offset = 128
    for _ in range(tensor_count):
        fields = RECORD.unpack_from(contents, offset)
        name_length, tensor_type, dimensions = fields[:3]
        shape = tuple(fields[3 : 3 + dimensions])
        data_bytes, auxiliary_bytes = fields[-2:]
        name_start = offset + RECORD.size
        name = contents[name_start : name_start + name_length].decode("ascii")
        data_start = aligned(name_start + name_length)
        auxiliary_start = data_start + data_bytes
        end = auxiliary_start + auxiliary_bytes
        if end > len(contents):
            raise ValueError(f"truncated tensor {name}")
        if tensor_type == TYPE_Q8_ROWWISE:
            values = np.frombuffer(
                contents, dtype="i1", count=data_bytes, offset=data_start
            ).copy().reshape(shape)
            scales = np.frombuffer(
                contents,
                dtype="<f4",
                count=auxiliary_bytes // 4,
                offset=auxiliary_start,
            ).copy()
            matrices[name] = (torch.from_numpy(values), torch.from_numpy(scales))
        elif tensor_type == TYPE_FLOAT32:
            values = np.frombuffer(
                contents,
                dtype="<f4",
                count=data_bytes // 4,
                offset=data_start,
            ).copy().reshape(shape)
            vectors[name] = torch.from_numpy(values)
        else:
            raise ValueError(f"unknown tensor type {tensor_type} for {name}")
        offset = aligned(end)
    if offset != len(contents):
        raise ValueError("unexpected trailing model bytes")
    return config, matrices, vectors


def load_exported_model(path: Path, device: torch.device) -> ThreeGSModel:
    config, matrices, vectors = parse_exported_model(path)
    model = ThreeGSModel(config)
    state = model.state_dict()
    for name, value in vectors.items():
        state[name].copy_(value)
    replace_quantized_matrices(model, matrices)
    return model.to(device).eval()
