"""Hybrid classical sampler network integrating QCNN feature extraction and a sampler head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumHybridSampler(nn.Module):
    """A hybrid network that combines QCNN-inspired feature extraction with a sampler head.
    The network processes an 8‑dimensional input, passes it through successive fully‑connected
    layers mirroring quantum convolution/pooling steps, then emits both a binary classification
    score and a 2‑class probability distribution used for sampling."""
    
    def __init__(self, input_dim: int = 8, seed: int | None = None) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh()
        )
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Heads
        self.class_head = nn.Linear(4, 1)
        self.sampler_head = nn.Sequential(nn.Linear(4, 2), nn.Softmax(dim=-1))
    
    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (classification, sampler distribution)."""
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        classification = torch.sigmoid(self.class_head(x))
        sampler_dist = self.sampler_head(x)
        return classification, sampler_dist

def SamplerQNN() -> QuantumHybridSampler:
    """Factory that mirrors the original SamplerQNN interface but returns the hybrid model."""
    return QuantumHybridSampler()

__all__ = ["QuantumHybridSampler", "SamplerQNN"]
