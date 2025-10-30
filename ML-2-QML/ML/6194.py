"""Advanced classical classifier mirroring the quantum helper interface."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumClassifierModel:
    """
    Classical feed‑forward network engineered for hybrid QML experiments.
    The architecture consists of *depth* residual blocks, each containing
    a linear layer, batch‑normalisation, ReLU activation and dropout.
    A final linear head maps to a binary output.

    The :meth:`build_classifier_circuit` static method returns:
        - ``network``: ``torch.nn.Sequential`` ready for training.
        - ``encoding``: list of feature indices (identity mapping).
        - ``weight_sizes``: number of learnable parameters per linear layer.
        - ``observables``: placeholder list of class indices (0, 1).

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of residual blocks in the body.

    Returns
    -------
    Tuple[nn.Module, List[int], List[int], List[int]]
        The classifier network and associated meta‑data.
    """

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int,
                                 dropout: float = 0.1,
                                 **kwargs) -> Tuple[nn.Module, List[int], List[int], List[int]]:
        # Residual block definition
        class ResidualBlock(nn.Module):
            def __init__(self, dim: int, dropout: float):
                super().__init__()
                self.linear = nn.Linear(dim, dim)
                self.bn = nn.BatchNorm1d(dim)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.linear(x)
                out = self.bn(out)
                out = F.relu(out)
                out = self.dropout(out)
                return out + x  # residual connection

        layers: List[nn.Module] = []
        in_dim = num_features

        # Initial projection
        layers.append(nn.Linear(in_dim, num_features))
        layers.append(nn.BatchNorm1d(num_features))
        layers.append(nn.ReLU())

        # Residual blocks
        for _ in range(depth):
            layers.append(ResidualBlock(num_features, dropout))

        # Output head
        head = nn.Linear(num_features, 2)
        layers.append(head)

        network = nn.Sequential(*layers)

        # He initialization for linear layers
        for m in network.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Metadata
        encoding = list(range(num_features))
        weight_sizes = [m.weight.numel() + m.bias.numel()
                        if hasattr(m, "weight") else 0
                        for m in network.modules()
                        if isinstance(m, nn.Linear)]
        observables = [0, 1]

        return network, encoding, weight_sizes, observables


__all__ = ["QuantumClassifierModel"]
