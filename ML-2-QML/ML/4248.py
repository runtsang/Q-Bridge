"""Hybrid classical model combining quanvolution, self‑attention, and a sampler network.

The model is fully compatible with the original ``Quanvolution`` interface but
extends it with a classical self‑attention block and a lightweight
sampler network.  The attention block operates on the flattened convolutional
features and the sampler injects a stochastic, data‑dependent bias that
mimics the behaviour of a quantum sampler.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalSelfAttention:
    """Simple self‑attention implementation used in the hybrid model."""

    def __init__(self, embed_dim: int) -> None:
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        # Project inputs to query/key/value spaces
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        # Compute attention scores
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


class SamplerModule(nn.Module):
    """Light‑weight sampler network that mimics a quantum sampler."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)


class QuanvolutionHybrid(nn.Module):
    """Classical hybrid model that fuses a quanvolution filter, attention and sampler."""

    def __init__(self, attention_dim: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.attention = ClassicalSelfAttention(embed_dim=attention_dim)
        self.sampler = SamplerModule()
        # Linear head: 4*14*14 (convolutional features) + attention_dim + 2 (sampler output)
        self.linear = nn.Linear(4 * 14 * 14 + attention_dim + 2, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Convolutional feature extraction
        conv_feat = self.conv(x)
        flat_feat = conv_feat.view(x.size(0), -1)

        # Random parameters for the attention block
        rot_params = np.random.randn(self.attention.embed_dim * 3)
        ent_params = np.random.randn(self.attention.embed_dim - 1)

        # Attention output
        attn_out = self.attention.run(
            rot_params, ent_params, flat_feat.cpu().numpy()
        )
        attn_tensor = torch.as_tensor(attn_out, dtype=torch.float32, device=x.device)

        # Sampler output (use first two feature columns)
        samp_input = flat_feat[:, :2]
        samp_out = self.sampler(samp_input)

        # Concatenate all signals
        combined = torch.cat([flat_feat, attn_tensor, samp_out], dim=1)
        logits = self.linear(combined)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
