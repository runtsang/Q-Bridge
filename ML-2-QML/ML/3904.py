"""
Hybrid QCNN combining classical convolutional layers with a quantum‑style estimator.
"""

from __future__ import annotations

import torch
from torch import nn

# --------------------------------------------------------------------------- #
#  Estimator: a small feed‑forward regressor that mimics the Qiskit EstimatorQNN
# --------------------------------------------------------------------------- #
class EstimatorNN(nn.Module):
    """Feed‑forward regressor used as a quantum‑style estimator."""
    def __init__(self, in_features: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.net(inputs)


# --------------------------------------------------------------------------- #
#  Hybrid QCNN: classical backbone + quantum‑style estimator
# --------------------------------------------------------------------------- #
class HybridQCNN(nn.Module):
    """Classical convolution‑inspired backbone followed by a quantum‑style estimator."""
    def __init__(self) -> None:
        super().__init__()
        # Classical feature extraction
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 6), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(6, 4), nn.Tanh())
        self.head = nn.Linear(4, 2)  # 2‑dimensional feature for the estimator

        # Quantum‑style estimator
        self.estimator = EstimatorNN(in_features=2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.head(x)                    # shape (..., 2)
        return self.estimator(x)            # final scalar output


def QCNN() -> HybridQCNN:
    """Factory returning the configured hybrid QCNN model."""
    return HybridQCNN()


__all__ = ["QCNN", "HybridQCNN", "EstimatorNN"]
