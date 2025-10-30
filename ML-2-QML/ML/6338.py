"""Hybrid classical convolutional‑fully‑connected model with optional quantum feature extraction.

The model extends the original Quantum‑NAT CNN by adding a residual block,
dropout, and a dedicated `encode_to_quantum_vector` method that produces a
compact representation suitable for quantum processing.  This design keeps the
classical component lightweight while exposing a clean interface for a
quantum module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFilter(nn.Module):
    """A lightweight, thresholded 2‑D convolutional filter.

    Parameters
    ----------
    kernel_size : int, default 3
        Size of the square kernel.
    threshold : float, default 0.0
        Activation threshold applied after the convolution.
    """

    def __init__(self, kernel_size: int = 3, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=(2, 3), keepdim=True)


class QFCHybridModel(nn.Module):
    """Classical CNN followed by a fully‑connected head with a quantum‑ready feature vector.

    The architecture mirrors the original Quantum‑NAT network but adds
    residual connections and dropout to improve generalisation.  A
    separate method, ``encode_to_quantum_vector``, returns a 16‑dimensional
    vector that can be fed to a quantum encoder.
    """

    def __init__(self) -> None:
        super().__init__()

        # Convolutional backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Residual block
        self.res_block = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
        )

        self.dropout = nn.Dropout2d(p=0.25)

        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(32 * 3 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        residual = self.res_block(x)
        x = F.relu(x + residual)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return self.norm(out)

    def encode_to_quantum_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Produce a 16‑dimensional vector for quantum processing.

        The vector is obtained by averaging over the spatial dimensions
        of a pooled intermediate feature map.
        """
        with torch.no_grad():
            pooled = F.avg_pool2d(x, kernel_size=6).view(x.shape[0], -1)
        return pooled


__all__ = ["QFCHybridModel"]
