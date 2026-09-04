#!/usr/bin/env python3
"""Generate replies from a training checkpoint for quality checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import ModelConfig, ThreeGSModel


def sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    generator: torch.Generator,
) -> int:
    logits = logits.float() / temperature
    if top_k > 0:
        count = min(top_k, logits.numel())
        threshold = torch.topk(logits, count).values[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    probabilities = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("prompt")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
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
    special = {
        name: tokenizer.token_to_id(f"<{name}>")
        for name in ("bos", "eos", "user", "assistant")
    }
    if any(identifier is None for identifier in special.values()):
        raise RuntimeError("tokenizer lacks a required special token")

    prompt_tokens = tokenizer.encode(
        args.prompt, add_special_tokens=False
    ).ids[-(config.context_length - 3) :]
    tokens = [
        special["bos"],
        special["user"],
        *prompt_tokens,
        special["assistant"],
    ]
    prefix_length = len(tokens)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            window = tokens[-config.context_length :]
            inputs = torch.tensor([window], dtype=torch.long, device=device)
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=device.type == "cuda",
            ):
                logits, _ = model(inputs)
            token = sample_token(
                logits[0, -1], args.temperature, args.top_k, generator
            )
            if token == special["eos"]:
                break
            tokens.append(token)

    answer_ids = tokens[prefix_length:]
    print(tokenizer.decode(answer_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
