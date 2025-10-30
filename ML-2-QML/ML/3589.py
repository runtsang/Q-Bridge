"""Hybrid estimator that fuses a classical feed-forward regression network with a quantum feature map.

The class EstimatorQNNGen219 accepts a feature size, a classical depth, a quantum depth, and an option to use a quantum layer.  When `use_quantum` is True the model expects the user to set the attribute `quantum_feature_map`, a callable that maps input tensors to a quantum expectation vector.  The final head is either a single neuron (regression) or a linear classifier (classification)."""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional, Iterable


class EstimatorQNNGen219(nn.Module):
    """Hybrid estimator that combines classical feed‑forward layers with a quantum feature extractor."""

    def __init__(
        self,
        num_features: int,
        classical_depth: int = 2,
        quantum_depth: int = 1,
        num_classes: int = 1,
        use_quantum: bool = True,
    ):
        """
        Parameters
        ----------
        num_features:
            Dimensionality of the input data.
        classical_depth:
            Number of hidden layers in the classical part.
        quantum_depth:
            Depth of the variational quantum circuit (used only if ``use_quantum`` is True).
        num_classes:
            Number of outputs. 1 → regression, >1 → classification.
        use_quantum:
            Flag to include quantum feature extraction.
        """
        super().__init__()
        self.num_features = num_features
        self.classical_depth = classical_depth
        self.quantum_depth = quantum_depth
        self.num_classes = num_classes
        self.use_quantum = use_quantum

        # Classical feature extractor
        layers: list[nn.Module] = []
        in_dim = num_features
        for _ in range(classical_depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.Tanh())
            in_dim = num_features
        self.classical_net = nn.Sequential(*layers)

        # Quantum feature extractor placeholder
        if use_quantum:
            self.quantum_dense = nn.Linear(num_features, num_features)
        else:
            self.quantum_dense = None

        # Final head
        if num_classes == 1:
            self.head = nn.Linear(num_features, 1)
        else:
            self.head = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid model.

        Parameters
        ----------
        x:
            Input tensor of shape (batch, num_features).
        """
        # Classical path
        x_cl = self.classical_net(x)

        # Quantum path
        if self.use_quantum and hasattr(self, "quantum_feature_map"):
            q_out = self.quantum_feature_map(x)
            q_out = self.quantum_dense(q_out)
            out = torch.cat([x_cl, q_out], dim=-1)
        else:
            out = x_cl

        # Final head
        return self.head(out)
