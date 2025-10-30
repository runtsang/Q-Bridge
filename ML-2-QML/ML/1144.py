"""Enhanced classical classifier with residual connections and dropout, mirroring the quantum interface."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """A single residual block with linear, batch‑norm, ReLU, dropout and skip."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=True)
        self.bn = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out + x

class QuantumClassifierModel:
    """Classical feed‑forward classifier with residual blocks and dropout.

    The API matches the quantum helper: ``build_classifier_circuit`` returns
    the network, a list of feature indices used for encoding, a list of
    parameter counts per layer, and the observable indices (output node
    indices).
    """

    @staticmethod
    def build_classifier_circuit(num_features: int,
                                 depth: int,
                                 dropout: float = 0.1
                                 ) -> Tuple[nn.Module,
                                            Iterable[int],
                                            List[int],
                                            List[int]]:
        """Construct a residual feed‑forward network.

        Parameters
        ----------
        num_features: int
            Number of input features (also the hidden dimension).
        depth: int
            Number of residual blocks.
        dropout: float, default 0.1
            Dropout probability applied after each activation.

        Returns
        -------
        network: nn.Module
            ``nn.Sequential`` containing the residual blocks and head.
        encoding: list[int]
            List of feature indices (0.. num_features-1).
        weight_sizes: list[int]
            Number of trainable parameters per layer (including biases).
        observables: list[int]
            Indices of the output logits (``[0, 1]``).
        """
        layers: List[nn.Module] = []
        weight_sizes: List[int] = []

        for _ in range(depth):
            block = ResidualBlock(num_features, dropout)
            layers.append(block)
            # Linear + BN parameters
            weight_sizes.append(block.linear.weight.numel() + block.linear.bias.numel())
            weight_sizes.append(block.bn.weight.numel() + block.bn.bias.numel())

        head = nn.Linear(num_features, 2, bias=True)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        network = nn.Sequential(*layers)
        encoding = list(range(num_features))
        observables = [0, 1]
        return network, encoding, weight_sizes, observables

__all__ = ["QuantumClassifierModel"]
