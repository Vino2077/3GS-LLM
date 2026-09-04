"""PyTorch reference for the exact transformer intended for the iPhone runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    context_length: int = 256
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    d_ff: int = 1024
    rope_base: float = 10_000.0
    rms_epsilon: float = 1e-5

    @property
    def head_dimension(self) -> int:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self.d_model // self.n_heads

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(
            value.float().pow(2).mean(dim=-1, keepdim=True) + self.epsilon
        )
        return (normalized * self.weight.float()).to(value.dtype)


def apply_rope(
    value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
) -> torch.Tensor:
    even = value[..., 0::2]
    odd = value[..., 1::2]
    rotated = torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    )
    return rotated.flatten(-2)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dimension = config.head_dimension
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self, value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
    ) -> torch.Tensor:
        batch, sequence, width = value.shape
        shape = (batch, sequence, self.n_heads, self.head_dimension)
        query = self.q_proj(value).view(shape).transpose(1, 2)
        key = self.k_proj(value).view(shape).transpose(1, 2)
        values = self.v_proj(value).view(shape).transpose(1, 2)
        query = apply_rope(query, cosine, sine)
        key = apply_rope(key, cosine, sine)
        output = F.scaled_dot_product_attention(
            query, key, values, dropout_p=0.0, is_causal=True
        )
        return self.o_proj(output.transpose(1, 2).contiguous().view(batch, sequence, width))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(value)) * self.up_proj(value))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.rms_epsilon)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_epsilon)
        self.feed_forward = FeedForward(config)

    def forward(
        self, value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
    ) -> torch.Tensor:
        value = value + self.attention(self.attention_norm(value), cosine, sine)
        return value + self.feed_forward(self.ffn_norm(value))


class ThreeGSModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.d_model, config.rms_epsilon)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        positions = torch.arange(config.context_length, dtype=torch.float32)
        inverse = 1.0 / (
            config.rope_base
            ** (
                torch.arange(0, config.head_dimension, 2, dtype=torch.float32)
                / config.head_dimension
            )
        )
        frequencies = torch.outer(positions, inverse)
        self.register_buffer("rope_cosine", frequencies.cos(), persistent=False)
        self.register_buffer("rope_sine", frequencies.sin(), persistent=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        sequence = tokens.shape[1]
        if sequence > self.config.context_length:
            raise ValueError("sequence exceeds configured context")
        value = self.token_embedding(tokens)
        cosine = self.rope_cosine[:sequence].to(value.dtype)[None, None, :, :]
        sine = self.rope_sine[:sequence].to(value.dtype)[None, None, :, :]
        for layer in self.layers:
            value = layer(value, cosine, sine)
        logits = self.lm_head(self.final_norm(value))
        loss = None
        if targets is not None:
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_targets = targets.reshape(-1)
            if loss_mask is None:
                loss = F.cross_entropy(flat_logits, flat_targets)
            else:
                token_losses = F.cross_entropy(
                    flat_logits, flat_targets, reduction="none"
                )
                flat_mask = loss_mask.reshape(-1).to(token_losses.dtype)
                loss = (token_losses * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)
        return logits, loss

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
