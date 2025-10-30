"""Hybrid classical model that emulates quantum‑inspired operations.

The architecture merges ideas from the original Quantum‑NAT CNN, a
quanvolutional 2×2 filter that produces four channels, and a fixed
random linear transformation that mimics a random quantum circuit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridQuantumNAT(nn.Module):
    """Classical neural network that emulates quantum‑inspired operations.

    The model consists of:
    - a 2×2 convolution producing four feature maps (quanvolution),
    - a frozen random linear layer that acts as a stand‑in for a
      random quantum circuit,
    - a fully‑connected head that maps the concatenated patch
      representations to class logits.
    """

    def __init__(self) -> None:
        super().__init__()
        # 2×2 convolution producing 4 channels (equivalent to a quanvolution filter)
        self.cnn = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # Fixed random linear layer (weights are frozen, no training)
        self.rand_lin = nn.Linear(4 * 14 * 14, 4 * 14 * 14, bias=False)
        nn.init.normal_(self.rand_lin.weight)
        self.rand_lin.weight.requires_grad = False
        self.register_buffer("rand_lin_weight", self.rand_lin.weight.clone().detach())
        # Fully‑connected head
        self.head = nn.Sequential(
            nn.Linear(4 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        self.norm = nn.BatchNorm1d(10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Extract 2×2 patches via stride‑2 convolution
        features = self.cnn(x)  # shape: (B, 4, 14, 14)
        flat = features.view(features.size(0), -1)  # (B, 4*14*14)
        # Apply the frozen random linear transformation
        flat = torch.matmul(flat, self.rand_lin_weight.t())
        flat = F.relu(flat)
        logits = self.head(flat)
        return self.norm(logits)


__all__ = ["HybridQuantumNAT"]
