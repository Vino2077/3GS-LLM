#!/usr/bin/env python3
"""Generate replies from a training checkpoint for quality checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import ModelConfig, ThreeGSModel
from sampling import PRESETS, SamplingPreset, generate_reply


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("prompt")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.02)
    parser.add_argument("--preset", choices=sorted(PRESETS))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    model = ThreeGSModel(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    preset = PRESETS[args.preset] if args.preset else SamplingPreset(
        "custom",
        args.temperature,
        args.top_k,
        args.repetition_penalty,
        greedy=args.top_k == 1,
    )
    result = generate_reply(
        model, tokenizer, args.prompt, preset, args.seed, args.max_new_tokens
    )
    print(result.text)


if __name__ == "__main__":
    main()
