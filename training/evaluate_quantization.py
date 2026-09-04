#!/usr/bin/env python3
"""Measure dynamic-activation/row-wise-weight INT8 loss before iOS export."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ModelConfig, ThreeGSModel
from train import TokenStream


def quantize_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scales = (weight.float().abs().amax(dim=1) / 127.0).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    values = torch.round(weight.float() / scales[:, None]).clamp(-127, 127)
    return values.to(torch.int8), scales


class QuantizedEmbedding(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        values, scales = quantize_rows(weight)
        self.register_buffer("values", values)
        self.register_buffer("scales", scales)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        values = F.embedding(tokens, self.values).float()
        scales = F.embedding(tokens, self.scales[:, None])
        return values * scales


class QuantizedLinear(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        values, scales = quantize_rows(weight)
        self.register_buffer("values", values)
        self.register_buffer("scales", scales)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        float_inputs = inputs.float()
        input_scales = (float_inputs.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        quantized = torch.round(float_inputs / input_scales).clamp(-127, 127)
        output = F.linear(quantized, self.values.float())
        return output * input_scales * self.scales


def quantize_model(model: ThreeGSModel) -> None:
    embedding_weight = model.token_embedding.weight.detach()
    model.token_embedding = QuantizedEmbedding(embedding_weight)
    model.lm_head = QuantizedLinear(embedding_weight)
    for layer in model.layers:
        layer.attention.q_proj = QuantizedLinear(layer.attention.q_proj.weight)
        layer.attention.k_proj = QuantizedLinear(layer.attention.k_proj.weight)
        layer.attention.v_proj = QuantizedLinear(layer.attention.v_proj.weight)
        layer.attention.o_proj = QuantizedLinear(layer.attention.o_proj.weight)
        layer.feed_forward.gate_proj = QuantizedLinear(
            layer.feed_forward.gate_proj.weight
        )
        layer.feed_forward.up_proj = QuantizedLinear(
            layer.feed_forward.up_proj.weight
        )
        layer.feed_forward.down_proj = QuantizedLinear(
            layer.feed_forward.down_proj.weight
        )


@torch.no_grad()
def evaluate(
    model: ThreeGSModel,
    data: Path,
    batch_size: int,
    batches: int,
    seed: int,
) -> tuple[float, float]:
    device = next(model.parameters()).device
    stream = TokenStream(
        data / "validation.bin",
        model.config.context_length,
        seed,
        data / "validation_response_mask.bin",
    )
    all_sum = 0.0
    response_sum = 0.0
    response_count = 0.0
    for _ in range(batches):
        inputs, targets, response_mask = stream.batch(batch_size, device)
        logits, _ = model(inputs)
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        )
        mask = response_mask.reshape(-1).float()
        all_sum += losses.sum().item()
        response_sum += (losses * mask).sum().item()
        response_count += mask.sum().item()
    token_count = batches * batch_size * model.config.context_length
    return all_sum / token_count, response_sum / response_count


def report(label: str, losses: tuple[float, float]) -> None:
    all_loss, response_loss = losses
    print(
        f"{label}: all_loss={all_loss:.5f} all_ppl={math.exp(all_loss):.2f} "
        f"response_loss={response_loss:.5f} response_ppl={math.exp(response_loss):.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = ThreeGSModel(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to("cuda").eval()
    baseline = evaluate(model, args.data, args.batch_size, args.batches, args.seed)
    report("FP32", baseline)
    quantize_model(model)
    quantized = evaluate(model, args.data, args.batch_size, args.batches, args.seed)
    report("INT8", quantized)
    print(
        f"delta: all={quantized[0] - baseline[0]:+.5f} "
        f"response={quantized[1] - baseline[1]:+.5f}"
    )


if __name__ == "__main__":
    main()
