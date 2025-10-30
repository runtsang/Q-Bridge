"""HybridQNAT: a hybrid classical-quantum model combining convolutional feature extraction and a quantum-inspired filter.

The model first extracts features via a small CNN (inspired by Quantum‑NAT), then applies a classical quantum filter
implemented as a thresholded convolution, and finally projects the concatenated representation to four outputs.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ClassicalFilter(nn.Module):
    """Thresholded convolutional filter mimicking the quantum quanvolution layer.

    The filter applies a 2×2 convolution followed by a sigmoid activation with a
    trainable bias. The output is the mean activation over the spatial map,
    producing a scalar per sample.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a scalar per batch element."""
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3], keepdim=True)


class HybridQNAT(nn.Module):
    """Hybrid classical model inspired by Conv and QuantumNAT.

    The architecture comprises:
        * A small CNN (mirroring QuantumNAT's feature extractor).
        * A classical quantum filter (ClassicalFilter) applied to the raw input.
        * A fully‑connected projection followed by batch‑norm.
    """

    def __init__(self) -> None:
        super().__init__()
        # Convolutional feature extractor (QuantumNAT style)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Classical quantum filter
        self.filter = ClassicalFilter(kernel_size=2, threshold=0.0)
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, H, W) with H=W=28 (MNIST‑like).

        Returns
        -------
        torch.Tensor
            Normalised 4‑dimensional output per sample.
        """
        bsz = x.shape[0]
        # CNN features
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        # Classical filter output
        filt_out = self.filter(x)  # shape (B, 1)
        # Concatenate
        concat = torch.cat([flat, filt_out], dim=1)
        out = self.fc(concat)
        return self.norm(out)


__all__ = ["HybridQNAT"]
