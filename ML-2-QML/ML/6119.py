"""Hybrid classical classifier that mirrors a quantum QCNN architecture.

The network consists of a feature‑map layer followed by a stack of
convolution‑plus‑pooling fully‑connected blocks.  The depth of the stack
can be tuned to match the depth of a quantum QCNN ansatz, allowing
direct comparison of parameter counts and expressibility.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Tuple

class QuantumClassifierModel(nn.Module):
    """Classical neural network mirroring a quantum QCNN classifier."""
    def __init__(self, num_features: int = 8, depth: int = 3) -> None:
        super().__init__()
        # Feature‑map layer (encodes raw features into a higher‑dimensional space)
        self.feature_map = nn.Sequential(nn.Linear(num_features, 16), nn.Tanh())

        # Build QCNN‑style blocks
        self.blocks: nn.ModuleList = nn.ModuleList()
        in_dim = 16
        for _ in range(depth):
            conv = nn.Sequential(nn.Linear(in_dim, in_dim), nn.Tanh())
            pool = nn.Sequential(nn.Linear(in_dim, in_dim // 2), nn.Tanh())
            self.blocks.append(nn.ModuleDict({"conv": conv, "pool": pool}))
            in_dim = in_dim // 2

        # Classification head
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        for block in self.blocks:
            x = block["conv"](x)
            x = block["pool"](x)
        return torch.sigmoid(self.head(x))

    def get_parameter_counts(self) -> Tuple[int, int]:
        """Return total number of trainable parameters and number of layers."""
        total = sum(p.numel() for p in self.parameters())
        layers = len(self.blocks) + 2  # feature map + head + blocks
        return total, layers

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Return a classical classifier network and metadata for parity with the quantum version."""
    net = QuantumClassifierModel(num_features, depth)
    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in net.parameters()]
    observables = [0]  # placeholder, no quantum observables
    return net, encoding, weight_sizes, observables
