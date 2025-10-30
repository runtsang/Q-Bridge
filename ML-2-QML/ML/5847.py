"""Hybrid QCNN implementation with extended classical feature extraction."""

from __future__ import annotations

import torch
from torch import nn

class ResidualBlock(nn.Module):
    """Basic residual block used in the classical feature extractor."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.act = nn.Tanh()
        self.residual = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(x) + self.act(self.linear(x))

class ClassicalFeatureExtractor(nn.Module):
    """Classical stack that mirrors the quantum convolution steps with residuals."""
    def __init__(self, in_features: int = 8, hidden: int = 32, out_features: int = 4) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh()
        )
        self.res1 = ResidualBlock(hidden)
        self.res2 = ResidualBlock(hidden)
        self.pool = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.pool(x)

class QCNNHybridModel(nn.Module):
    """Hybrid classical–quantum model. The quantum part is represented by a placeholder
    function that must be provided externally. In this pure‑classical version we
    simply perform a weighted sum of the classical embeddings."""
    def __init__(self, weight: float = 0.5) -> None:
        super().__init__()
        self.classical = ClassicalFeatureExtractor()
        self.classifier = nn.Linear(4, 1)
        self.weight = weight  # weight for blending with quantum output

    def forward(self, x: torch.Tensor, quantum_output: torch.Tensor | None = None) -> torch.Tensor:
        cls_feat = self.classical(x)
        cls_out = torch.sigmoid(self.classifier(cls_feat))
        if quantum_output is None:
            return cls_out
        return self.weight * cls_out + (1 - self.weight) * torch.sigmoid(quantum_output)

def QCNN() -> QCNNHybridModel:
    """Factory returning the configured :class:`QCNNHybridModel`."""
    return QCNNHybridModel()

__all__ = ["QCNNHybridModel", "QCNN"]
