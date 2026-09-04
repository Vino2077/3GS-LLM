#!/usr/bin/env python3
"""Generate a reproducible human-readable quality report for checkpoints."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import ModelConfig, ThreeGSModel
from sampling import PRESETS, generate_reply, repeated_ngram_rate


def stable_seed(base: int, index: int) -> int:
    return (base + index * 1_000_003) & 0x7FFF_FFFF


def topic_hit(text: str, terms: list[str]) -> bool:
    folded = text.casefold().replace("ё", "е")
    return any(term.casefold().replace("ё", "е") in folded for term in terms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("prompts", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--presets", nargs="+", default=["legacy", "A", "B", "C", "greedy"]
    )
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    unknown = [name for name in args.presets if name not in PRESETS]
    if unknown:
        parser.error(f"unknown presets {unknown}; choices: {sorted(PRESETS)}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = ThreeGSModel(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or not 30 <= len(prompts) <= 50:
        raise ValueError("eval prompt set must contain 30-50 entries")

    lines = [
        f"checkpoint: {args.checkpoint}",
        f"checkpoint_step: {checkpoint['step']}",
        f"seed: {args.seed}",
        f"max_new_tokens: {args.max_new_tokens}",
        "",
    ]
    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "presets": {},
    }
    for preset_name in args.presets:
        preset = PRESETS[preset_name]
        lengths: list[int] = []
        repetitions: list[float] = []
        eos_count = 0
        hit_count = 0
        category_hits: dict[str, list[bool]] = defaultdict(list)
        lines.extend(
            [
                "=" * 78,
                (
                    f"PRESET {preset.name}: temperature={preset.temperature:g} "
                    f"top_k={preset.top_k} repetition_penalty="
                    f"{preset.repetition_penalty:g}"
                ),
                "=" * 78,
                "",
            ]
        )
        for index, item in enumerate(prompts):
            result = generate_reply(
                model,
                tokenizer,
                item["prompt"],
                preset,
                stable_seed(args.seed, index),
                args.max_new_tokens,
            )
            terms = item.get("expected_topic", [])
            hit = topic_hit(result.text, terms) if terms else False
            lengths.append(len(result.token_ids))
            repetitions.append(repeated_ngram_rate(result.token_ids))
            eos_count += int(result.reached_eos)
            hit_count += int(hit)
            category_hits[item["category"]].append(hit)
            lines.extend(
                [
                    f"[{item['id']}] {item['category']}",
                    f"USER: {item['prompt']}",
                    f"MODEL: {result.text}",
                    (
                        f"META: tokens={len(result.token_ids)} "
                        f"eos={result.reached_eos} topic_term_hit={hit}"
                    ),
                    "",
                ]
            )
        count = len(prompts)
        metrics = {
            "prompts": count,
            "average_response_tokens": sum(lengths) / count,
            "eos_rate": eos_count / count,
            "repeated_3gram_rate": sum(repetitions) / count,
            "topic_term_hit_rate": hit_count / count,
            "category_topic_term_hit_rate": {
                category: sum(values) / len(values)
                for category, values in sorted(category_hits.items())
            },
        }
        summary["presets"][preset_name] = metrics  # type: ignore[index]
        lines.extend(
            [
                "METRICS: "
                + " ".join(
                    [
                        f"avg_tokens={metrics['average_response_tokens']:.2f}",
                        f"eos={metrics['eos_rate']:.1%}",
                        f"repeat3={metrics['repeated_3gram_rate']:.2%}",
                        f"topic_terms={metrics['topic_term_hit_rate']:.1%}",
                    ]
                ),
                "",
            ]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output} and {summary_path}")


if __name__ == "__main__":
    main()
