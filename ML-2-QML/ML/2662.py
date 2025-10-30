"""Classical QCNN-inspired network with depth‑adaptive layers and optional classifier."""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Tuple, List

class QCNNGen226(nn.Module):
    """Depth‑controlled classical convolutional network mimicking the quantum QCNN.

    The architecture follows the same stacking pattern as the quantum version:
    a feature‑map layer, followed by alternating convolution and pooling stages.
    The number of conv/pool pairs is governed by *depth*, allowing a smooth
    scaling from shallow to deep models.  Each stage uses a linear transform
    followed by a nonlinear activation; pooling is implemented as a linear
    reduction that halves the dimensionality, mirroring the qubit‑reduction
    performed by the quantum pooling layers.
    """

    def __init__(self, num_features: int = 8, depth: int = 3, hidden_multiplier: int = 2) -> None:
        super().__init__()
        self.depth = depth

        # Feature‑map: expand the raw input to a richer representation.
        self.feature_map = nn.Sequential(
            nn.Linear(num_features, hidden_multiplier * num_features),
            nn.Tanh()
        )

        # Build alternating convolution / pooling stages.
        layers: List[nn.Module] = []
        in_dim = hidden_multiplier * num_features
        for stage in range(depth):
            conv = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.Tanh()
            )
            pool = nn.Sequential(
                nn.Linear(in_dim, in_dim // 2),
                nn.Tanh()
            )
            layers.extend([conv, pool])
            in_dim = in_dim // 2

        # Final classifier head.
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.body(x)
        return torch.sigmoid(self.head(x))

def build_classifier_circuit(
    num_features: int, depth: int
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Create a feed‑forward classifier that mirrors the quantum helper.

    Parameters
    ----------
    num_features : int
        Input dimensionality.
    depth : int
        Number of hidden layers.

    Returns
    -------
    nn.Module
        The constructed network.
    Iterable[int]
        Indices of the input encoding (simply ``range(num_features)``).
    Iterable[int]
        Flattened list of parameter counts per layer.
    List[int]
        Observables indices used by the quantum counterpart.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

__all__ = ["QCNNGen226", "build_classifier_circuit"]
