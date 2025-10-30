from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QCNNFeatureExtractor(nn.Module):
    """QCNN‐inspired feature extractor using fully connected layers."""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 16):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.pool1  = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh())
        self.layer3 = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.Tanh())
        self.pool2  = nn.Sequential(nn.Linear(hidden_dim // 4, hidden_dim // 8), nn.Tanh())
        self.layer4 = nn.Sequential(nn.Linear(hidden_dim // 8, hidden_dim // 8), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool1(x)
        x = self.layer3(x)
        x = self.pool2(x)
        x = self.layer4(x)
        return x

class RBFKernel(nn.Module):
    """Gaussian kernel with learnable gamma."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class HybridSamplerQNN(nn.Module):
    """
    Combines a QCNN feature extractor, RBF kernel similarity to learnable prototypes,
    and a softmax sampler that outputs a 2‑class probability distribution.
    """
    def __init__(self, feature_dim: int = 8, prototype_count: int = 4):
        super().__init__()
        self.extractor = QCNNFeatureExtractor(input_dim=2, hidden_dim=feature_dim)
        self.kernel     = RBFKernel(gamma=1.0)
        self.prototypes = nn.Parameter(torch.randn(prototype_count, feature_dim))
        self.sampler    = nn.Sequential(
            nn.Linear(prototype_count, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extractor(x)
        sims = [self.kernel(features, p) for p in self.prototypes]
        sims = torch.cat(sims, dim=-1)
        logits = self.sampler(sims)
        return F.softmax(logits, dim=-1)

def HybridSamplerQNNFactory() -> HybridSamplerQNN:
    """Factory that returns a configured HybridSamplerQNN."""
    return HybridSamplerQNN(feature_dim=8, prototype_count=4)

__all__ = ["HybridSamplerQNN", "HybridSamplerQNNFactory"]
