#!/usr/bin/env python3
"""Measure dynamic-activation/row-wise-weight INT8 loss before iOS export."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from int8_reference import quantize_model
from model import ModelConfig, ThreeGSModel
from train import create_stream


@torch.no_grad()
def evaluate(
    model: ThreeGSModel,
    data: Path,
    batch_size: int,
    batches: int,
    seed: int,
    aligned_pairs: bool,
) -> tuple[float, float]:
    device = next(model.parameters()).device
    stream = create_stream(
        data,
        "validation",
        model.config.context_length,
        seed,
        aligned_pairs,
    )
    all_sum = 0.0
    response_sum = 0.0
    response_count = 0.0
    all_count = 0.0
    for _ in range(batches):
        inputs, targets, response_mask = stream.batch(batch_size, device)
        logits, _ = model(inputs)
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        )
        mask = response_mask.reshape(-1).float()
        valid = targets.reshape(-1).ne(0).float()
        all_sum += (losses * valid).sum().item()
        all_count += valid.sum().item()
        response_sum += (losses * mask).sum().item()
        response_count += mask.sum().item()
    return all_sum / all_count, response_sum / response_count


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
    parser.add_argument("--aligned-pairs", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = ThreeGSModel(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to("cuda").eval()
    baseline = evaluate(
        model,
        args.data,
        args.batch_size,
        args.batches,
        args.seed,
        args.aligned_pairs,
    )
    report("FP32", baseline)
    quantize_model(model)
    quantized = evaluate(
        model,
        args.data,
        args.batch_size,
        args.batches,
        args.seed,
        args.aligned_pairs,
    )
    report("INT8", quantized)
    print(
        f"delta: all={quantized[0] - baseline[0]:+.5f} "
        f"response={quantized[1] - baseline[1]:+.5f}"
    )


if __name__ == "__main__":
    main()
