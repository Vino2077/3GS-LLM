#!/usr/bin/env python3
"""Quick CPU test for model shape, gradients, and weight tying."""

from __future__ import annotations

import torch

from model import ModelConfig, ThreeGSModel


def main() -> None:
    config = ModelConfig(
        vocab_size=64,
        context_length=16,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
    )
    model = ThreeGSModel(config)
    tokens = torch.randint(0, config.vocab_size, (2, config.context_length))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (2, config.context_length, config.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()

    response_mask = torch.zeros_like(tokens)
    response_mask[:, -4:] = 1
    _, masked_loss = model(tokens, tokens, response_mask)
    assert masked_loss is not None and torch.isfinite(masked_loss)

    production = ThreeGSModel(ModelConfig())
    expected = 17_308_032
    actual = production.parameter_count()
    if actual != expected:
        raise AssertionError(f"production parameter count {actual:,} != {expected:,}")
    print(f"smoke test passed; production model has {actual:,} parameters")


if __name__ == "__main__":
    main()
