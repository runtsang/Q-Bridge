"""Hybrid classical sampler network combining convolutional feature extraction and quantum-inspired sampling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def HybridSamplerQNN():
    """Factory returning a hybrid sampler network."""
    class HybridSamplerQNNClass(nn.Module):
        """Hybrid sampler with convolutional feature extraction and softmax output."""

        def __init__(self, input_dim: int = 8, hidden_dim: int = 16, num_classes: int = 2) -> None:
            super().__init__()
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
                nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.Tanh(),
            )
            self.sampler = nn.Linear(hidden_dim // 4, num_classes)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = self.feature_extractor(inputs)
            logits = self.sampler(x)
            return F.softmax(logits, dim=-1)

    return HybridSamplerQNNClass()


__all__ = ["HybridSamplerQNN"]
