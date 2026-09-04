#!/usr/bin/env python3
"""Check deterministic presets and the C-compatible repetition penalty."""

from __future__ import annotations

import torch

from sampling import PRESETS, apply_repetition_penalty, sample_token


def main() -> None:
    logits = torch.tensor([0.0, 2.0, 5.0, -2.0])
    adjusted = apply_repetition_penalty(logits, [2, 3], 2.0)
    assert torch.equal(adjusted, torch.tensor([0.0, 2.0, 2.5, -4.0]))
    first = torch.Generator().manual_seed(123)
    second = torch.Generator().manual_seed(123)
    history: list[int] = []
    sequence_one = [
        sample_token(logits, history, PRESETS["A"], first, set())
        for _ in range(20)
    ]
    sequence_two = [
        sample_token(logits, history, PRESETS["A"], second, set())
        for _ in range(20)
    ]
    assert sequence_one == sequence_two
    assert sample_token(logits, history, PRESETS["greedy"], first, set()) == 2
    assert set(("legacy", "A", "B", "C", "near-greedy", "greedy")) <= PRESETS.keys()
    print("sampling smoke test passed")


if __name__ == "__main__":
    main()
