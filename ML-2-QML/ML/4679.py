from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridKernelHead(nn.Module):
    """
    Classical surrogate for a quantum expectation head.
    Computes an RBF kernel between the network features and a set of
    trainable support vectors, then applies a linear projection and a
    differentiable sigmoid activation with an optional shift.
    """
    def __init__(
        self,
        in_features: int,
        support_vectors: int = 12,
        gamma: float = 0.5,
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        # Trainable support vectors (k, d)
        self.support = nn.Parameter(torch.randn(support_vectors, in_features))
        self.gamma = gamma
        self.shift = shift
        self.linear = nn.Linear(support_vectors, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (batch, d)
        Returns a probability tensor of shape (batch, 1).
        """
        # Compute pairwise squared Euclidean distances
        diff = features.unsqueeze(1) - self.support.unsqueeze(0)  # (b, k, d)
        sq = (diff * diff).sum(-1)  # (b, k)
        kernel = torch.exp(-self.gamma * sq)  # (b, k)
        logits = self.linear(kernel)  # (b, 1)
        return torch.sigmoid(logits + self.shift)

class QuantumNATHybrid(nn.Module):
    """
    Classical CNN + kernel‑based head that mirrors the structure of the
    Quantum‑NAT model.  The output is a binary probability vector
    [p, 1‑p] suitable for cross‑entropy loss.
    """
    def __init__(self) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )
        # Fully‑connected projection
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        # Kernel‑based hybrid head
        self.head = HybridKernelHead(
            in_features=32, support_vectors=12, gamma=0.5, shift=0.0
        )
        self.norm = nn.BatchNorm1d(32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass producing a two‑class probability vector.
        """
        x = self.features(x)  # (b, 16, 7, 7)
        x = torch.flatten(x, 1)  # (b, 16*7*7)
        x = self.fc(x)  # (b, 32)
        p = self.head(x)  # (b, 1)
        return torch.cat((p, 1 - p), dim=-1)

__all__ = ["QuantumNATHybrid"]
