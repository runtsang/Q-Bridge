"""Hybrid CNN + feed‑forward regressor combining EstimatorQNN and QuantumNAT architectures.

The network extracts image features using a small convolutional backbone, projects them
into a four‑dimensional feature space, and then applies a linear regression head.
The design mirrors the classical EstimatorQNN feed‑forward block while inheriting the
spatial feature extraction of QuantumNAT’s QFCModel."""
from __future__ import annotations

import torch
from torch import nn


class EstimatorQNNGen197(nn.Module):
    """CNN‑based regressor with a 2‑layer fully‑connected head.

    The architecture is a light‑weight variant of QFCModel: a 2‑layer conv‑pool‑conv‑pool
    backbone producing 16×7×7 activations, which are flattened and linearly projected
    to a 4‑dimensional embedding.  A final ``Linear`` layer maps this embedding to a
    scalar regression target, optionally followed by a hyperbolic tangent to bound the
    output.

    The model is fully differentiable and can be trained with standard PyTorch
    optimizers.
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
        )
        # 16×7×7 features → 64 → 4
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.BatchNorm1d(4),
        )
        # Final regression head
        self.out = nn.Sequential(
            nn.Linear(4, 1),
            nn.Tanh(),  # optional bounding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Expect input shape [batch, 1, 28, 28] (e.g., MNIST)
        features = self.features(x)
        flattened = features.view(x.shape[0], -1)
        embedding = self.fc(flattened)
        return self.out(embedding)


__all__ = ["EstimatorQNNGen197"]
