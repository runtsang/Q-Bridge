from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSamplerClassifier(nn.Module):
    """
    Classical hybrid sampler‑classifier.

    * Feature extractor: lightweight CNN (2 conv layers + pooling) inspired by QuantumNAT.
    * Feature projection: linear layer reducing to `feature_dim` units.
    * Sampler head: two‑layer feed‑forward network producing class probabilities.
    * Depth hyper‑parameter controls hidden unit count in the sampler head.
    """

    def __init__(
        self,
        image_channels: int = 1,
        image_size: int = 28,
        feature_dim: int = 64,
        sampler_hidden: int = 4,
        depth: int = 1,
    ) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(image_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Compute flattened size
        dummy = torch.zeros(1, image_channels, image_size, image_size)
        feat_size = self.features(dummy).view(1, -1).size(1)
        # Feature projection
        self.fc = nn.Sequential(nn.Linear(feat_size, feature_dim), nn.ReLU())
        # Sampler head
        self.sampler = nn.Sequential(
            nn.Linear(feature_dim, sampler_hidden),
            nn.Tanh(),
            nn.Linear(sampler_hidden, 2),
        )
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: image -> features -> sampler logits -> softmax.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        logits = self.sampler(x)
        return F.softmax(logits, dim=-1)


__all__ = ["HybridSamplerClassifier"]
