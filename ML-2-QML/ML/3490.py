"""Hybrid classical model combining the Quantum‑NAT CNN backbone with a QCNN‑style fully‑connected stack."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridQuantumNet(nn.Module):
    """
    Classical hybrid network that mirrors the architecture of Quantum‑NAT and QCNN.

    The model consists of:
        * A 2‑D convolutional backbone (3 conv layers + 2× max‑pooling) identical to
          the convolutional part of the original QFCModel.
        * A fully‑connected stack inspired by the QCNNModel (feature map + conv/pool
          linear layers) that processes the flattened features.
        * Batch‑norm output for stability.
    """

    def __init__(self) -> None:
        super().__init__()

        # Convolutional backbone (3 conv layers + 2 max‑pooling)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 14×14
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 7×7
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Flattened dimension after 7×7 conv
        flat_dim = 32 * 7 * 7

        # QCNN‑style linear stack
        self.fc_stack = nn.Sequential(
            nn.Linear(flat_dim, 64), nn.Tanh(),
            nn.Linear(64, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 16), nn.Tanh(),
            nn.Linear(16, 4)
        )

        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, 1, H, W)``.
        """
        features = self.backbone(x)
        flat = features.view(features.size(0), -1)
        out = self.fc_stack(flat)
        return self.norm(out)


def HybridQuantumNetFactory() -> HybridQuantumNet:
    """Convenience factory that returns a new ``HybridQuantumNet`` instance."""
    return HybridQuantumNet()


__all__ = ["HybridQuantumNet", "HybridQuantumNetFactory"]
