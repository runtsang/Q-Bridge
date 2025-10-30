"""Hybrid classical classifier combining data‑encoding and photonic‑style layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import torch
import torch.nn as nn

@dataclass
class FraudLikeLayerParams:
    """Parameters for a single photonic‑style layer."""
    weight: torch.Tensor
    bias: torch.Tensor
    scale: torch.Tensor
    shift: torch.Tensor
    clip: bool = False

class FraudLikeLayer(nn.Module):
    """Linear ➜ Tanh ➜ affine rescale, with optional clipping."""
    def __init__(self, params: FraudLikeLayerParams) -> None:
        super().__init__()
        self.linear = nn.Linear(params.weight.shape[1], params.weight.shape[0])
        with torch.no_grad():
            self.linear.weight.copy_(params.weight)
            self.linear.bias.copy_(params.bias)
        self.activation = nn.Tanh()
        self.register_buffer("scale", params.scale)
        self.register_buffer("shift", params.shift)
        self.clip = params.clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.activation(self.linear(x))
        out = out * self.scale + self.shift
        return out

def _clip_tensor(t: torch.Tensor, bound: float) -> torch.Tensor:
    return torch.clamp(t, -bound, bound)

def build_classifier_circuit(num_features: int, depth: int, clip_weights: bool = False) -> Tuple[nn.Module, Iterable[int], List[int], List[int]]:
    """
    Build a hybrid PyTorch network that mirrors the quantum data‑encoding and
    fraud‑detection style layers.  The first layer is un‑clipped; subsequent
    layers are optionally clipped to keep parameters bounded.
    """
    layers: List[nn.Module] = []
    weight_sizes: List[int] = []

    # Initial encoding layer – simple linear embedding
    in_dim = num_features
    out_dim = num_features
    w = torch.randn(out_dim, in_dim)
    b = torch.randn(out_dim)
    scale = torch.ones(out_dim)
    shift = torch.zeros(out_dim)
    params = FraudLikeLayerParams(w, b, scale, shift, clip=False)
    layers.append(FraudLikeLayer(params))
    weight_sizes.append((w.numel() + b.numel()))

    # Variational layers
    for _ in range(depth):
        w = torch.randn(out_dim, in_dim)
        b = torch.randn(out_dim)
        scale = torch.randn(out_dim) * 0.1
        shift = torch.randn(out_dim) * 0.1
        if clip_weights:
            w = _clip_tensor(w, 5.0)
            b = _clip_tensor(b, 5.0)
        params = FraudLikeLayerParams(w, b, scale, shift, clip=clip_weights)
        layers.append(FraudLikeLayer(params))
        weight_sizes.append((w.numel() + b.numel()))

    # Final classifier head
    head = nn.Linear(out_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    encoding = list(range(num_features))
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

__all__ = ["build_classifier_circuit", "FraudLikeLayer", "FraudLikeLayerParams"]
