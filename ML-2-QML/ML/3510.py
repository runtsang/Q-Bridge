"""Hybrid QCNN combining classical convolutional layers and a quantum ansatz."""

from __future__ import annotations

import torch
from torch import nn
from.qml_qcnn import QCNNQuantum, SamplerQNN  # quantum components in the same package

class QCNNHybrid(nn.Module):
    """Hybrid architecture that processes data classically before feeding it to a QCNN ansatz."""

    def __init__(self, input_dim: int = 8) -> None:
        super().__init__()
        # Classical feature extractor
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
        )
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 8), nn.Tanh())

        # Quantum block
        self.quantum = QCNNQuantum()
        # Final classification head
        self.classifier = nn.Linear(1, 1)

        # Optional sampler head
        self.sampler = SamplerQNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hybrid network."""
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        # Quantum evaluation expects a 2‑D tensor of shape (batch, 8)
        quantum_out = self.quantum(x)
        logits = self.classifier(quantum_out)
        return torch.sigmoid(logits)

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """Return sampling probabilities from the quantum sampler head."""
        return self.sampler(x)

__all__ = ["QCNNHybrid"]
