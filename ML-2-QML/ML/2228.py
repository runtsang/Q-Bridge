"""Hybrid sampler network combining quanvolution and classical sampler."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """2‑D convolutional filter that mimics a quantum kernel over 2×2 patches."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.conv(x).view(x.size(0), -1)


class SamplerModule(nn.Module):
    """Simple two‑output sampler network."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)


class HybridSamplerQNN(nn.Module):
    """
    Classical hybrid network: quanvolution filter → linear reduction → sampler.
    Produces a probability distribution over two classes.
    """
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.reduce = nn.Linear(4 * 14 * 14, 2)
        self.sampler = SamplerModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        reduced = self.reduce(features)
        return self.sampler(reduced)


__all__ = ["HybridSamplerQNN"]
