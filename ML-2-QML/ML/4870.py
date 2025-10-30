"""Hybrid fraud detection model – classical implementation.

The ML side expands the original photonic analogue with:
* a 2‑D Conv filter (from Conv.py) to capture local patterns,
* a deep feed‑forward network with batch‑norm, dropout and sigmoid output,
* concatenation of the conv feature with the raw input.

The class is fully compatible with PyTorch training loops.
"""

from __future__ import annotations

import torch
from torch import nn
from.Conv import Conv  # assumes Conv.py in the same package


class FraudDetectionHybrid(nn.Module):
    """A PyTorch model that emulates the photonic fraud‑detection circuit
    while adding modern regularisation and a convolutional pre‑processor.
    """

    def __init__(
        self,
        num_features: int = 2,
        hidden_sizes: Sequence[int] = (64, 32),
        depth: int = 2,
        kernel_size: int = 2,
        conv_threshold: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        num_features:
            Dimensionality of the raw input vector.
        hidden_sizes:
            Sizes of the hidden layers in the feed‑forward part.
        depth:
            Depth used for the quantum classifier counterpart
            (kept for API symmetry).
        kernel_size:
            Size of the 2×2 convolutional patch.
        conv_threshold:
            Threshold used by the Conv filter.
        """
        super().__init__()
        self.preconv = Conv(kernel_size=kernel_size, threshold=conv_threshold)

        layers = []
        in_dim = num_features + 1  # +1 for conv scalar
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.3))
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def _conv_feature(self, batch: torch.Tensor) -> torch.Tensor:
        """Apply the classical Conv filter to each sample in the batch."""
        conv_vals = []
        for sample in batch:
            # Conv expects a 2‑D array; we reshape the feature vector.
            arr = sample.cpu().numpy().reshape(1, 1, 1, -1)
            conv_vals.append(self.preconv.run(arr))
        return torch.tensor(conv_vals, dtype=torch.float32, device=batch.device).unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Tensor of shape (batch, num_features).

        Returns
        -------
        Tensor
            Fraud probability of shape (batch, 1).
        """
        conv_feat = self._conv_feature(x)
        x_cat = torch.cat([x, conv_feat], dim=1)
        return self.net(x_cat)


__all__ = ["FraudDetectionHybrid"]
