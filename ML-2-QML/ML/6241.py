"""Hybrid classical estimator combining a lightweight MLP with optional CNN feature extraction.

The model can process either flat 2‑dimensional inputs (as in the original EstimatorQNN)
or image‑like tensors by first passing them through a small CNN (inspired by Quantum‑NAT).
The final regression head is a shallow MLP that outputs a single value.
"""

import torch
from torch import nn

class HybridEstimator(nn.Module):
    """Hybrid neural network that unifies a simple MLP with optional CNN feature extraction."""

    def __init__(self, input_shape: tuple[int,...], use_cnn: bool = False) -> None:
        """
        Parameters
        ----------
        input_shape
            Shape of a single sample, e.g. ``(2,)`` for 2‑D vectors or ``(1,28,28)`` for grayscale images.
        use_cnn
            When ``True`` a shallow CNN (Conv‑ReLU‑Pool‑Conv‑ReLU‑Pool) extracts spatial features
            before the regression head.  Defaults to ``False`` for vector inputs.
        """
        super().__init__()
        self.use_cnn = use_cnn

        if use_cnn:
            # Inspired by Quantum‑NAT's QFCModel
            self.features = nn.Sequential(
                nn.Conv2d(input_shape[0], 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            # Compute flattened feature size
            dummy = torch.zeros(1, *input_shape)
            feat = self.features(dummy)
            flat_features = feat.numel()
            self.fc = nn.Sequential(
                nn.Linear(flat_features, 64),
                nn.ReLU(),
                nn.Linear(64, 4),
                nn.ReLU(),
                nn.Linear(4, 1),
            )
        else:
            # Plain MLP for flat vectors
            self.net = nn.Sequential(
                nn.Linear(input_shape[0], 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.use_cnn:
            features = self.features(x)
            flattened = features.view(features.size(0), -1)
            return self.fc(flattened)
        return self.net(x)

__all__ = ["HybridEstimator"]
