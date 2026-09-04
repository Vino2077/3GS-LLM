#!/usr/bin/env python3
"""Compare checkpoints on identical validation blocks and both loss modes."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

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
    all_loss_sum = 0.0
    response_loss_sum = 0.0
    response_count = 0.0
    all_count = 0.0
    model.eval()
    for _ in range(batches):
        inputs, targets, response_mask = stream.batch(batch_size, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(inputs)
        losses = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        )
        valid = targets.reshape(-1).ne(0).float()
        all_loss_sum += (losses * valid).sum().item()
        all_count += valid.sum().item()
        mask = response_mask.reshape(-1).float()
        response_loss_sum += (losses * mask).sum().item()
        response_count += mask.sum().item()
    return all_loss_sum / all_count, response_loss_sum / response_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--aligned-pairs",
        action="store_true",
        help="evaluate complete pairs using *_offsets.bin instead of random windows",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda")
    model = ThreeGSModel(ModelConfig()).to(device)
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["config"] != model.config.to_dict():
            raise ValueError(f"model configuration mismatch: {path}")
        model.load_state_dict(checkpoint["model"])
        all_loss, response_loss = evaluate(
            model,
            args.data,
            args.batch_size,
            args.batches,
            args.seed,
            args.aligned_pairs,
        )
        print(
            f"step={int(checkpoint['step']):5d} "
            f"all_loss={all_loss:.5f} all_ppl={math.exp(all_loss):.2f} "
            f"response_loss={response_loss:.5f} "
            f"response_ppl={math.exp(response_loss):.2f} path={path}"
        )


if __name__ == "__main__":
    main()
