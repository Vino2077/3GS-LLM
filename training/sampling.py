"""Shared deterministic generation and sampling presets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenizers import Tokenizer


@dataclass(frozen=True)
class SamplingPreset:
    name: str
    temperature: float
    top_k: int
    repetition_penalty: float
    greedy: bool = False


PRESETS = {
    "legacy": SamplingPreset("legacy", 0.80, 40, 1.08),
    "A": SamplingPreset("A", 0.45, 10, 1.00),
    "B": SamplingPreset("B", 0.25, 5, 1.00),
    "C": SamplingPreset("C", 0.60, 20, 1.02),
    "near-greedy": SamplingPreset("near-greedy", 0.10, 3, 1.00),
    "greedy": SamplingPreset("greedy", 1.00, 1, 1.00, greedy=True),
}

BANNED_SPECIAL_NAMES = ("pad", "bos", "user", "assistant", "unk")


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    reached_eos: bool


def special_token_ids(tokenizer: Tokenizer) -> dict[str, int]:
    result: dict[str, int] = {}
    for name in ("pad", "bos", "eos", "user", "assistant", "unk"):
        identifier = tokenizer.token_to_id(f"<{name}>")
        if identifier is None:
            raise RuntimeError(f"tokenizer lacks <{name}>")
        result[name] = identifier
    return result


def apply_repetition_penalty(
    logits: torch.Tensor,
    history: list[int],
    penalty: float,
    window: int = 64,
) -> torch.Tensor:
    if penalty <= 0.0:
        raise ValueError("repetition penalty must be positive")
    if penalty == 1.0 or not history:
        return logits
    adjusted = logits.clone()
    seen = torch.tensor(
        sorted(set(history[-window:])), dtype=torch.long, device=logits.device
    )
    values = adjusted[seen]
    adjusted[seen] = torch.where(values >= 0.0, values / penalty, values * penalty)
    return adjusted


def sample_token(
    logits: torch.Tensor,
    history: list[int],
    preset: SamplingPreset,
    generator: torch.Generator,
    banned_ids: set[int],
) -> int:
    scores = apply_repetition_penalty(
        logits.float(), history, preset.repetition_penalty
    )
    if banned_ids:
        indices = torch.tensor(sorted(banned_ids), device=scores.device)
        scores[indices] = float("-inf")
    if preset.greedy or preset.top_k == 1:
        return int(torch.argmax(scores).item())
    if preset.temperature <= 0.0:
        raise ValueError("temperature must be positive")
    count = min(preset.top_k, scores.numel())
    values, indices = torch.topk(scores, count, sorted=True)
    probabilities = torch.softmax(values / preset.temperature, dim=-1)
    selected = torch.multinomial(probabilities, 1, generator=generator)
    return int(indices[selected].item())


@torch.inference_mode()
def generate_reply(
    model,
    tokenizer: Tokenizer,
    prompt: str,
    preset: SamplingPreset,
    seed: int,
    max_new_tokens: int = 64,
) -> GenerationResult:
    ids = special_token_ids(tokenizer)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    prompt_ids = prompt_ids[-(model.config.context_length - 3) :]
    history = [ids["bos"], ids["user"], *prompt_ids, ids["assistant"]]
    prefix_length = len(history)
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    banned = {ids[name] for name in BANNED_SPECIAL_NAMES}
    reached_eos = False

    for _ in range(max_new_tokens):
        window = history[-model.config.context_length :]
        inputs = torch.tensor([window], dtype=torch.long, device=device)
        autocast_enabled = device.type == "cuda" and any(
            parameter.dtype in (torch.float16, torch.bfloat16)
            for parameter in model.parameters()
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            logits, _ = model(inputs)
        token = sample_token(
            logits[0, -1], history, preset, generator, banned
        )
        if token == ids["eos"]:
            reached_eos = True
            break
        history.append(token)

    response_ids = history[prefix_length:]
    return GenerationResult(
        text=tokenizer.decode(response_ids, skip_special_tokens=True),
        token_ids=response_ids,
        reached_eos=reached_eos,
    )


def repeated_ngram_rate(tokens: list[int], order: int = 3) -> float:
    if len(tokens) < order:
        return 0.0
    ngrams = [
        tuple(tokens[index : index + order])
        for index in range(len(tokens) - order + 1)
    ]
    return 1.0 - len(set(ngrams)) / len(ngrams)
