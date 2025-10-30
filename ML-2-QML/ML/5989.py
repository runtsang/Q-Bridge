from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSamplerQNN(nn.Module):
    """Classical hybrid sampler network combining CNN feature extraction and softmax output."""
    def __init__(self) -> None:
        super().__init__()
        # Convolutional backbone inspired by QFCModel
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully connected head and batch‑norm
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.encoder(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        out = self.norm(out)
        return self.softmax(out)

__all__ = ["HybridSamplerQNN"]
