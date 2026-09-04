#!/usr/bin/env python3
"""Train 3GS-LM-17M from a packed uint16 token stream."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import ModelConfig, ThreeGSModel


class TokenStream:
    def __init__(self, path: Path, context: int, seed: int, mask_path: Path) -> None:
        self.tokens = np.memmap(path, mode="r", dtype="<u2")
        self.response_mask = np.memmap(mask_path, mode="r", dtype="u1")
        if len(self.tokens) != len(self.response_mask):
            raise ValueError(f"token/mask length mismatch: {path}")
        if len(self.tokens) <= context + 1:
            raise ValueError(f"token stream is too short: {path}")
        self.context = context
        self.rng = np.random.default_rng(seed)

    def batch(
        self, size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        starts = self.rng.integers(0, len(self.tokens) - self.context - 1, size=size)
        samples = np.stack(
            [self.tokens[start : start + self.context + 1] for start in starts]
        ).astype(np.int64, copy=False)
        masks = np.stack(
            [self.response_mask[start + 1 : start + self.context + 1] for start in starts]
        ).astype(np.uint8, copy=False)
        tensor = torch.from_numpy(samples).to(device, non_blocking=True)
        mask_tensor = torch.from_numpy(masks).to(device, non_blocking=True)
        return tensor[:, :-1], tensor[:, 1:], mask_tensor


@torch.no_grad()
def evaluate(
    model: ThreeGSModel,
    stream: TokenStream,
    batch_size: int,
    batches: int,
    device: torch.device,
    autocast,
    response_only: bool,
) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        inputs, targets, response_mask = stream.batch(batch_size, device)
        with autocast():
            _, loss = model(
                inputs, targets, response_mask if response_only else None
            )
        assert loss is not None
        losses.append(loss.float())
    model.train()
    return torch.stack(losses).mean().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--response-only",
        action="store_true",
        help="compute loss only on assistant response and EOS tokens",
    )
    parser.add_argument(
        "--initial-weights",
        type=Path,
        help="checkpoint whose model weights initialize this training stage",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the full training run")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    config = ModelConfig()
    model = ThreeGSModel(config).to(device)
    if args.initial_weights:
        initial = torch.load(args.initial_weights, map_location="cpu", weights_only=True)
        if initial.get("config") != config.to_dict():
            raise ValueError("initial checkpoint uses a different model configuration")
        model.load_state_dict(initial["model"])
        print(f"loaded initial model weights from {args.initial_weights}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )
    train_stream = TokenStream(
        args.data / "train.bin",
        config.context_length,
        args.seed,
        args.data / "train_response_mask.bin",
    )
    validation_stream = TokenStream(
        args.data / "validation.bin",
        config.context_length,
        args.seed + 1,
        args.data / "validation_response_mask.bin",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8"
    )

    use_bfloat16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    autocast = lambda: torch.autocast(device_type="cuda", dtype=dtype)
    print(
        f"training {model.parameter_count():,} parameters on "
        f"{torch.cuda.get_device_name(0)} with {dtype}; "
        f"loss={'responses' if args.response_only else 'all tokens'}"
    )

    tokens_per_step = (
        args.batch_size * config.context_length * args.gradient_accumulation
    )
    started = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        if step <= args.warmup_steps:
            learning_rate = args.learning_rate * step / args.warmup_steps
        else:
            progress = (step - args.warmup_steps) / max(
                1, args.steps - args.warmup_steps
            )
            coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
            learning_rate = args.min_learning_rate + coefficient * (
                args.learning_rate - args.min_learning_rate
            )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(args.gradient_accumulation):
            inputs, targets, response_mask = train_stream.batch(
                args.batch_size, device
            )
            with autocast():
                _, loss = model(
                    inputs,
                    targets,
                    response_mask if args.response_only else None,
                )
                assert loss is not None
                scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()
            accumulated_loss += loss.detach().float().item()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 10 == 0:
            elapsed = time.perf_counter() - started
            processed = step * tokens_per_step
            print(
                f"step={step:5d} loss={accumulated_loss / args.gradient_accumulation:.4f} "
                f"lr={learning_rate:.2e} grad={float(gradient_norm):.3f} "
                f"tokens/s={processed / elapsed:,.0f}"
            )

        if step % args.eval_interval == 0 or step == args.steps:
            validation_loss = evaluate(
                model,
                validation_stream,
                args.batch_size,
                args.eval_batches,
                device,
                autocast,
                args.response_only,
            )
            print(f"validation step={step} loss={validation_loss:.4f}")

        if step % args.save_interval == 0 or step == args.steps:
            checkpoint = {
                "step": step,
                "config": config.to_dict(),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "response_only": args.response_only,
            }
            temporary = args.output / "checkpoint.tmp"
            final = args.output / f"checkpoint-{step:05d}.pt"
            torch.save(checkpoint, temporary)
            temporary.replace(final)
            print(f"saved {final}")


if __name__ == "__main__":
    main()
