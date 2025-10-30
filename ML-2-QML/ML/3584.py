"""Hybrid classical estimator that fuses CNN feature extraction with a fully‑connected regression head.

The architecture is inspired by the EstimatorQNN and QuantumNAT examples.  It accepts a 2‑channel
image (or a 1×1 image with 2 channels), extracts spatial features with a short convolutional
backbone, flattens the result, and applies a sequence of linear layers with tanh activations
leading to a single‑output regression.  A batch‑norm layer normalises the final output.  The
model is fully differentiable and does not rely on any quantum libraries.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

def EstimatorQNN() -> nn.Module:
    """Return a hybrid CNN‑FC regression model."""
    class _Estimator(nn.Module):
        def __init__(self, in_channels: int = 2, num_outputs: int = 1, dropout: float = 0.0) -> None:
            super().__init__()
            # Convolutional feature extractor (QuantumNAT style)
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            # Placeholder for flattened feature size; set on first forward pass
            self._flatten = None
            # Fully‑connected head mirroring EstimatorQNN widths
            self.fc = nn.Sequential(
                nn.Linear(64, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, num_outputs),
            )
            self.norm = nn.BatchNorm1d(num_outputs)
            self.dropout = nn.Dropout(dropout)

        def _init_flatten(self, x: torch.Tensor) -> None:
            """Determine the flattened size after the feature extractor."""
            with torch.no_grad():
                out = self.features(x)
                self._flatten = out.view(out.size(0), -1).size(1)
                # Re‑create the first linear layer with the correct in‑features
                self.fc[0] = nn.Linear(self._flatten, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.

            Accepts either a (B, C, H, W) tensor or a (B, C) tensor which is reshaped to
            (B, C, 1, 1).  The network is agnostic to the exact spatial resolution
            as long as the convolutional layers can process it.
            """
            if x.dim() == 2:
                # Treat as (B, C) and reshape
                x = x.view(-1, x.size(1), 1, 1)
            if self._flatten is None:
                self._init_flatten(x)
            feat = self.features(x)
            flat = feat.view(feat.size(0), -1)
            out = self.fc(flat)
            out = self.dropout(out)
            return self.norm(out)

    return _Estimator()
__all__ = ["EstimatorQNN"]
