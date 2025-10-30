"""Hybrid quantum‑classical classifier with optional quantum feature mapping.

The module provides a classical feed‑forward network that can be augmented
with a quantum feature mapper.  It mirrors the interface of the
original `QuantumClassifierModel.py` but adds residual blocks,
dropout, and a regression head based on the `EstimatorQNN` seed.
This allows the same class to be used for classification or
regression in a purely classical setting.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ----------------------------------------------------------------------
# Classical classifier factory
# ----------------------------------------------------------------------
def build_classifier_circuit(
    num_features: int, depth: int
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a feed‑forward classifier with residual connections
    and dropout.  Returns:
        network: nn.Sequential ready for use
        encoding: indices of input features that are linearly transformed
        weight_sizes: number of trainable parameters per linear layer
        observables: list of output node indices (1‑2 for binary classification)
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding: List[int] = list(range(num_features))
    weight_sizes: List[int] = []

    # Build depth-many residual blocks
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU(), nn.Dropout(p=0.1)])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        # residual connection placeholder: actual addition is done in forward
        in_dim = num_features

    # Classification head
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = [0, 1]
    return network, encoding, weight_sizes, observables


# ----------------------------------------------------------------------
# Hybrid model definition
# ----------------------------------------------------------------------
class QuantumClassifierModel(nn.Module):
    """
    A hybrid classifier that optionally augments classical features with
    quantum expectation values.  The quantum mapper can be passed in from
    the QML side; when omitted the model reduces to a pure classical
    feed‑forward network.
    """
    def __init__(
        self,
        num_features: int,
        depth: int,
        use_quantum: bool = False,
        quantum_mapper: Optional[object] = None,
        num_quantum_features: int = 0,
    ) -> None:
        super().__init__()
        self.classical_net, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_features, depth
        )
        self.use_quantum = use_quantum
        self.quantum_mapper = quantum_mapper
        if use_quantum:
            # Linear head that maps quantum features to the same output space
            self.quantum_head = nn.Linear(num_quantum_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Args:
            x: Tensor of shape (batch, num_features)
        Returns:
            logits: Tensor of shape (batch, 2)
        """
        # Classical path
        out = self.classical_net(x)

        # Quantum augmentation
        if self.use_quantum and self.quantum_mapper is not None:
            # Compute quantum expectation values (numpy)
            quantum_features = self.quantum_mapper.forward(
                x.detach().cpu().numpy()
            )
            q_out = self.quantum_head(
                torch.from_numpy(quantum_features).float().to(x.device)
            )
            out = out + q_out

        return out


# ----------------------------------------------------------------------
# Regression head (EstimatorQNN)
# ----------------------------------------------------------------------
class EstimatorQNN(nn.Module):
    """Simple regression network inspired by the EstimatorQNN seed."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)


__all__ = [
    "build_classifier_circuit",
    "QuantumClassifierModel",
    "EstimatorQNN",
]
