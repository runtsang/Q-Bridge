"""Pure classical hybrid model with CNN encoder and fully connected projection.

The class UnifiedFCQuantumHybrid implements a classical CNN encoder inspired by
the QuantumNAT architecture, followed by a linear projection to the dimensionality
of the quantum device.  The output is batch‑normalized and can be used as
logits for classification tasks.  The module is fully PyTorch‑based and
does not depend on any quantum libraries.

The API matches the quantum variant: a ``run`` method accepts a batch of
inputs and returns a tensor of shape (batch, output_dim).  The class can be
instantiated with ``UnifiedFCQuantumHybrid()`` or via the factory
``UnifiedFCQuantumHybrid.factory()``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Iterable, Tuple

class UnifiedFCQuantumHybrid(nn.Module):
    """Classical CNN encoder + linear projection.

    The architecture mirrors the encoder in QuantumNAT and the fully
    connected projection in QCNN.  The model is fully differentiable
    and can be trained with standard PyTorch optimisers.
    """
    def __init__(self, in_channels: int = 1, num_qubits: int = 4,
                 out_features: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        # Encoder: two Conv layers + pooling
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Projection to match quantum device dimension
        # Assume input after pooling has shape (bsz, 16, 7, 7) for 28x28 images
        self.projection = nn.Sequential(
            nn.Linear(16 * 7 * 7, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_qubits),
        )
        self.batch_norm = nn.BatchNorm1d(num_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classical encoder."""
        bsz = x.shape[0]
        feats = self.encoder(x)
        feats = feats.view(bsz, -1)
        out = self.projection(feats)
        out = self.batch_norm(out)
        return out

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper matching the quantum variant."""
        return self.forward(x)

    @classmethod
    def factory(cls) -> "UnifiedFCQuantumHybrid":
        """Return a ready‑to‑use instance."""
        return cls()


__all__ = ["UnifiedFCQuantumHybrid"]
