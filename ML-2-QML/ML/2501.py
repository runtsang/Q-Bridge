"""Hybrid QCNN model combining convolutional structure with photonic-inspired parameterization.

The classical model mirrors the quantum QCNN architecture but replaces the raw
weight matrices with compact photonic parameter blocks (beam‑splitter angles,
squeezing, displacement and Kerr phases).  Each block is mapped to a 2×2
linear layer followed by a Tanh activation, a scale and a shift – exactly
as in the photonic fraud‑detection example.  The resulting network can be
trained with standard optimisers while still reflecting the structure of a
QCNN, enabling an end‑to‑end comparison between the classical and quantum
implementations.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, List

@dataclass
class PhotonicParams:
    """Compact representation of a 2‑mode photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: PhotonicParams, *, clip: bool) -> nn.Module:
    # Build a 2×2 linear transformation from the beam‑splitter parameters.
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()

class QCNNHybrid(nn.Module):
    """Classical analogue of a QCNN with photonic‑inspired layers.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    num_conv_layers : int
        Number of convolutional blocks to apply.
    num_pool_layers : int
        Number of pooling blocks to apply.
    photonic_params : Iterable[PhotonicParams]
        Sequence of parameters for the photonic layers that replace the
        standard fully‑connected layers in the QCNN.
    """

    def __init__(
        self,
        num_features: int,
        num_conv_layers: int,
        num_pool_layers: int,
        photonic_params: Iterable[PhotonicParams],
    ) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(num_features, 16), nn.Tanh())
        self.conv_layers: List[nn.Module] = nn.ModuleList()
        self.pool_layers: List[nn.Module] = nn.ModuleList()
        # Build convolutional blocks
        for i in range(num_conv_layers):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Linear(16, 16), nn.Tanh()
                )
            )
        # Build pooling blocks
        for i in range(num_pool_layers):
            self.pool_layers.append(
                nn.Sequential(
                    nn.Linear(16, 12), nn.Tanh()
                )
            )
        # Photonic-inspired layers
        self.photonic_blocks = nn.ModuleList(
            [_layer_from_params(p, clip=True) for p in photonic_params]
        )
        self.head = nn.Linear(12, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        for conv in self.conv_layers:
            x = conv(x)
        for pool in self.pool_layers:
            x = pool(x)
        for block in self.photonic_blocks:
            x = block(x)
        return torch.sigmoid(self.head(x))

def QCNNHybridFactory(
    num_features: int = 8,
    num_conv_layers: int = 3,
    num_pool_layers: int = 3,
    photonic_params: Iterable[PhotonicParams] | None = None,
) -> QCNNHybrid:
    """Return a fully configured ``QCNNHybrid`` instance."""
    if photonic_params is None:
        # Default to a single trivial photonic block
        default = PhotonicParams(
            bs_theta=0.0,
            bs_phi=0.0,
            phases=(0.0, 0.0),
            squeeze_r=(0.0, 0.0),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(1.0, 1.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        photonic_params = [default]
    return QCNNHybrid(num_features, num_conv_layers, num_pool_layers, photonic_params)

__all__ = ["PhotonicParams", "QCNNHybrid", "QCNNHybridFactory"]
