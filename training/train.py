#!/usr/bin/env python3
"""Train 3GS-LM-17M from packed windows or complete aligned pairs."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import ModelConfig, ThreeGSModel
from conditional_loss import ranking_loss


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


class AlignedPairStream:
    """Random batches of complete BOS/USER/ASSISTANT/EOS pairs."""

    def __init__(
        self,
        path: Path,
        context: int,
        seed: int,
        mask_path: Path,
        offsets_path: Path,
        pad_token_id: int = 0,
    ) -> None:
        self.tokens = np.memmap(path, mode="r", dtype="<u2")
        self.response_mask = np.memmap(mask_path, mode="r", dtype="u1")
        self.offsets = np.memmap(offsets_path, mode="r", dtype="<u8")
        if len(self.tokens) != len(self.response_mask):
            raise ValueError(f"token/mask length mismatch: {path}")
        if len(self.offsets) < 2 or self.offsets[0] != 0:
            raise ValueError(f"invalid aligned-pair offsets: {offsets_path}")
        if int(self.offsets[-1]) != len(self.tokens):
            raise ValueError(f"offsets do not cover token stream: {offsets_path}")
        lengths = np.diff(self.offsets)
        if int(lengths.max()) > context or int(lengths.min()) < 5:
            raise ValueError(f"aligned sample length is invalid: {offsets_path}")
        self.context = context
        self.pad_token_id = pad_token_id
        self.rng = np.random.default_rng(seed)

    def batch(
        self, size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = self.rng.integers(0, len(self.offsets) - 1, size=size)
        ranges = [
            (int(self.offsets[index]), int(self.offsets[index + 1]))
            for index in indices
        ]
        width = max(end - start for start, end in ranges)
        samples = np.full((size, width), self.pad_token_id, dtype=np.int64)
        masks = np.zeros((size, width - 1), dtype=np.uint8)
        for row, (start, end) in enumerate(ranges):
            length = end - start
            samples[row, :length] = self.tokens[start:end]
            masks[row, : length - 1] = self.response_mask[start + 1 : end]
        tensor = torch.from_numpy(samples).to(device, non_blocking=True)
        mask_tensor = torch.from_numpy(masks).to(device, non_blocking=True)
        return tensor[:, :-1], tensor[:, 1:], mask_tensor


def create_stream(
    data: Path,
    split: str,
    context: int,
    seed: int,
    aligned_pairs: bool,
) -> TokenStream | AlignedPairStream:
    token_path = data / f"{split}.bin"
    mask_path = data / f"{split}_response_mask.bin"
    if not aligned_pairs:
        return TokenStream(token_path, context, seed, mask_path)
    offsets_path = data / f"{split}_offsets.bin"
    if not offsets_path.exists():
        raise RuntimeError(
            f"{offsets_path} is missing; rerun tokenize_dataset.py so "
            "response-only training can use complete aligned pairs"
        )
    metadata_path = data / "dataset_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pad_token_id = int(metadata.get("special_tokens", {}).get("pad", 0))
    return AlignedPairStream(
        token_path,
        context,
        seed,
        mask_path,
        offsets_path,
        pad_token_id,
    )


def prefix_loss_weights(mask: torch.Tensor, targets: torch.Tensor,
                        prefix_tokens: int, prefix_weight: float) -> torch.Tensor:
    """Weight only the first response tokens, never USER, padding or EOS (id 2)."""
    weights = mask.float()
    prefix = (mask.cumsum(dim=1) <= prefix_tokens) & mask.bool() & (targets != 2)
    return torch.where(prefix, weights * prefix_weight, weights)


@torch.no_grad()
def evaluate(
    model: ThreeGSModel,
    stream: TokenStream | AlignedPairStream,
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
    parser.add_argument("--response-prefix-tokens", type=int, default=16)
    parser.add_argument("--response-prefix-weight", type=float, default=1.0)
    parser.add_argument("--allow-strong-prefix-experiment", action="store_true",
                        help="explicitly allow weights up to 8 for a short response-start pilot")
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-margin", type=float, default=0.25)
    parser.add_argument(
        "--response-only",
        action="store_true",
        help=(
            "train on complete aligned pairs and compute loss only on assistant "
            "response and EOS tokens"
        ),
    )
    parser.add_argument(
        "--initial-weights",
        type=Path,
        help="checkpoint whose model weights initialize this training stage",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume model and optimizer state; --steps remains the final step",
    )
    args = parser.parse_args()
    if args.initial_weights and args.resume:
        parser.error("--initial-weights and --resume are mutually exclusive")
    maximum_weight = 8 if args.allow_strong_prefix_experiment else 2
    if not 1 <= args.response_prefix_weight <= maximum_weight or not 0 <= args.response_prefix_tokens <= 24:
        parser.error("prefix weight must be 1..2 (explicit experiment: up to 8), prefix tokens 0..24")
    if args.response_prefix_weight != 1 and not args.response_only:
        parser.error("prefix weighting requires --response-only")
    if not 0 <= args.contrastive_weight <= .5 or args.contrastive_margin < 0:
        parser.error("contrastive weight must be 0..0.5 and margin nonnegative")
    if args.contrastive_weight and (not args.response_only or args.batch_size < 2):
        parser.error("contrastive experiment requires aligned batches of at least 2")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the full training run")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    config = ModelConfig()
    model = ThreeGSModel(config).to(device)
    start_step = 0
    initial_path = args.resume or args.initial_weights
    initial = None
    if initial_path:
        initial = torch.load(initial_path, map_location="cpu", weights_only=True)
        if initial.get("config") != config.to_dict():
            raise ValueError("initial checkpoint uses a different model configuration")
        model.load_state_dict(initial["model"])
        print(f"loaded initial model weights from {initial_path}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )
    if args.resume:
        assert initial is not None
        if bool(initial.get("response_only")) != args.response_only:
            raise ValueError("resume checkpoint uses a different loss mode")
        expected_mode = "aligned_pairs" if args.response_only else "packed_stream"
        saved_mode = initial.get("data_mode")
        if saved_mode is None:
            saved_mode = (
                "legacy_random_windows" if args.response_only else "packed_stream"
            )
        if saved_mode != expected_mode:
            raise ValueError("resume checkpoint uses a different data mode")
        if initial.get("response_prefix_weight", 1.0) != args.response_prefix_weight or initial.get("response_prefix_tokens", 16) != args.response_prefix_tokens:
            raise ValueError("resume checkpoint uses different prefix weighting")
        for key, default in (("contrastive_weight", 0.0), ("contrastive_margin", .25)):
            if initial.get("training_args", {}).get(key, default) != getattr(args, key):
                raise ValueError("resume checkpoint uses different contrastive settings")
        optimizer.load_state_dict(initial["optimizer"])
        start_step = int(initial["step"])
        if start_step >= args.steps:
            raise ValueError("resume checkpoint has already reached --steps")
        print(f"resuming optimizer at step {start_step}")
    train_stream = create_stream(
        args.data,
        "train",
        config.context_length,
        args.seed,
        args.response_only,
    )
    validation_stream = create_stream(
        args.data,
        "validation",
        config.context_length,
        args.seed + 1,
        args.response_only,
    )
    if args.resume and initial is not None and "rng_state" in initial:
        train_stream.rng.bit_generator.state = initial["rng_state"]["train"]
        validation_stream.rng.bit_generator.state = initial["rng_state"]["validation"]
        torch.set_rng_state(initial["rng_state"]["torch"])
        torch.cuda.set_rng_state_all(initial["rng_state"]["cuda"])
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
    for step in range(start_step + 1, args.steps + 1):
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
                loss_weights = prefix_loss_weights(response_mask, targets,
                    args.response_prefix_tokens, args.response_prefix_weight) if args.response_only else None
                logits, loss = model(
                    inputs,
                    targets,
                    loss_weights,
                )
                assert loss is not None
                if args.contrastive_weight:
                    loss = loss + args.contrastive_weight * ranking_loss(
                        model, logits, inputs, targets, response_mask, args.contrastive_margin)
                scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()
            accumulated_loss += loss.detach().float().item()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 10 == 0:
            elapsed = time.perf_counter() - started
            processed = (step - start_step) * tokens_per_step
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
                "data_mode": "aligned_pairs" if args.response_only else "packed_stream",
                "response_prefix_weight": args.response_prefix_weight,
                "response_prefix_tokens": args.response_prefix_tokens,
                "training_args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "rng_state": {"train": train_stream.rng.bit_generator.state,
                    "validation": validation_stream.rng.bit_generator.state,
                    "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
            }
            temporary = args.output / "checkpoint.tmp"
            final = args.output / f"checkpoint-{step:05d}.pt"
            torch.save(checkpoint, temporary)
            temporary.replace(final)
            print(f"saved {final}")


if __name__ == "__main__":
    main()
