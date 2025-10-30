"""Hybrid classical model combining convolutional feature extraction with a quantum‑inspired sampler.

The architecture extends the baseline QFCModel by adding residual connections, dropout,
and a 4‑dimensional probability sampler.  The sampler is a lightweight neural network
that maps the 4‑dimensional feature vector to a probability distribution, mimicking
the behaviour of the quantum SamplerQNN while remaining fully classical.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSampler(nn.Module):
    """4‑dimensional probability sampler inspired by the original SamplerQNN."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

class HybridNat(nn.Module):
    """Classical hybrid network: CNN → FC → Sampler."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4),
        )
        self.norm = nn.BatchNorm1d(4)
        self.sampler = HybridSampler()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        out = self.norm(out)
        probs = self.sampler(out)
        return probs

__all__ = ["HybridNat"]
