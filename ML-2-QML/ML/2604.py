"""Hybrid classical classifier combining Qiskit circuit structure
and photonic fraud‑detection scaling/ clipping.

The network mirrors the quantum ansatz depth and feature size.
Each linear layer optionally clips weights/biases to [-5,5]
and applies a per‑layer scale/shift after the ReLU, emulating
the displacement operation in the photonic model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

class _ScaleShift(nn.Module):
    """Utility module implementing per‑layer scale and shift."""
    def __init__(self, scale: torch.Tensor, shift: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("shift", shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift

@dataclass
class LayerParams:
    """Per‑layer hyper‑parameters for clipping and post‑activation scaling."""
    clip: bool = False
    scale: float = 1.0
    shift: float = 0.0

class QuantumClassifierModel:
    """Builds a classical feed‑forward network that follows the same
    topology as the quantum circuit defined in the companion QML module.
    """

    def __init__(self, num_features: int, depth: int,
                 *, clip: bool = False, scale: float = 1.0, shift: float = 0.0) -> None:
        self.num_features = num_features
        self.depth = depth
        self.clip = clip
        self.scale = scale
        self.shift = shift
        self.network = self._build_network()

    def _build_network(self) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_dim = self.num_features
        for _ in range(self.depth):
            linear = nn.Linear(in_dim, self.num_features)
            if self.clip:
                with torch.no_grad():
                    linear.weight.clamp_(-5.0, 5.0)
                    linear.bias.clamp_(-5.0, 5.0)
            layers.append(linear)
            layers.append(nn.ReLU())
            if self.scale!= 1.0 or self.shift!= 0.0:
                scale_tensor = torch.full((self.num_features,), self.scale, dtype=torch.float32)
                shift_tensor = torch.full((self.num_features,), self.shift, dtype=torch.float32)
                layers.append(_ScaleShift(scale_tensor, shift_tensor))
            in_dim = self.num_features
        head = nn.Linear(in_dim, 2)
        if self.clip:
            with torch.no_grad():
                head.weight.clamp_(-5.0, 5.0)
                head.bias.clamp_(-5.0, 5.0)
        layers.append(head)
        return nn.Sequential(*layers)

    def weight_sizes(self) -> List[int]:
        """Return the number of trainable parameters per linear layer."""
        return [m.weight.numel() + m.bias.numel()
                for m in self.network.modules() if isinstance(m, nn.Linear)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def encode_data(self, x: torch.Tensor) -> torch.Tensor:
        """Identity encoding – data is fed directly to the network."""
        return x

__all__ = ["QuantumClassifierModel"]
