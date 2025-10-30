"""Hybrid classical convolutional model that emulates a quantum filter and adds a CNN+FC backbone.

The design merges the drop‑in `Conv` filter with the `QFCModel` architecture:
* A first 2×2 convolutional layer with a learnable bias and sigmoid activation
  that mimics the quantum filter.
* Two standard convolution–ReLU–pool blocks identical to the Quantum‑NAT
  feature extractor.
* A fully connected head producing four outputs, followed by batch‑norm.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvHybrid(nn.Module):
    """Hybrid classical convolutional model."""
    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        in_channels: int = 1,
        out_channels: int = 8,
    ) -> None:
        super().__init__()
        # Quantum‑filter analogue: 2‑by‑2 conv + bias + sigmoid
        self.quantum_filter = nn.Conv2d(
            in_channels, 1, kernel_size=kernel_size, bias=True
        )
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=False)

        # Classical feature extractor (Quantum‑NAT style)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Fully connected head
        self.classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Apply the quantum‑filter analogue
        qf = self.quantum_filter(x)
        qf = torch.sigmoid(qf - self.threshold)
        # Pass through the rest of the network
        out = self.features(qf)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return self.bn(out)

    def run(self, data: torch.Tensor | torch.Tensor) -> torch.Tensor:
        """Convenience wrapper that accepts raw torch tensor or numpy array."""
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=torch.float32)
        return self.forward(data)
