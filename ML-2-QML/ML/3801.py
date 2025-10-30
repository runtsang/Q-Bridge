"""Hybrid classical convolutional model that emulates the Quantum‑NAT architecture.

The implementation follows the same feature extraction pipeline as the original
QFCModel but replaces the quantum block with a classical surrogate.  This makes
the model fully trainable on a CPU/GPU without any quantum simulator while
preserving the expressive capacity of the original architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridConvModel(nn.Module):
    """Classical hybrid model mirroring the Quantum‑NAT CNN + FC architecture.

    Attributes
    ----------
    feature_extractor : nn.Sequential
        Two convolutional layers with ReLU activations and max‑pooling.
    proj : nn.Linear
        Projects the flattened feature map to the 64‑dimensional space used
        by the quantum fully‑connected block in the original paper.
    quantum_surrogate : nn.Linear
        Trainable linear layer initialized with orthogonal weights to
        emulate the unitary rotations of the quantum fully‑connected block.
    norm : nn.BatchNorm1d
        Normalises the final four‑dimensional output.
    """

    def __init__(self,
                 kernel_size: int = 3,
                 conv_channels: int = 8,
                 num_classes: int = 4,
                 seed: int | None = None) -> None:
        super().__init__()

        # Feature extractor mimicking the first two conv layers of QFCModel
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=kernel_size,
                      stride=1, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_channels, conv_channels * 2,
                      kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Compute the output size after two pooling operations on a 28×28 image
        dummy_input = torch.zeros(1, 1, 28, 28)
        dummy_feat = self.feature_extractor(dummy_input)
        flat_dim = dummy_feat.view(1, -1).size(1)

        # Projection to match the 64‑dimensional space used by the quantum block
        self.proj = nn.Linear(flat_dim, 64)

        # Surrogate for the quantum fully‑connected layer
        torch.manual_seed(seed)
        self.quantum_surrogate = nn.Linear(64, num_classes)
        nn.init.orthogonal_(self.quantum_surrogate.weight)

        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.feature_extractor(x)
        flattened = features.view(bsz, -1)
        projected = self.proj(flattened)
        out = self.quantum_surrogate(projected)
        return self.norm(out)

__all__ = ["HybridConvModel"]
