"""Combined classical estimator with hybrid architecture inspired by EstimatorQNN and QuantumClassifierModel.

The class supports both regression and classification tasks. It merges the deep feed‑forward
structure of the classical estimator with the depth‑controlled ReLU/Tanh alternation
from the classifier prototype.  The network exposes the same metadata interface
(encoding indices and weight sizes) that the quantum side expects, enabling a
straight‑forward interface for hybrid training pipelines.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import List

class EstimatorQNNGen042(nn.Module):
    """
    A configurable feed‑forward network that can act as a regressor or a binary classifier.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input data.
    depth : int, default 3
        Number of hidden layers.  Each hidden layer is followed by a non‑linearity
        that alternates between Tanh (regression) and ReLU (classification).
    task : str, {"regression", "classification"}, default "regression"
        The downstream objective.  For classification the network outputs two logits.
    """

    def __init__(self, num_features: int, depth: int = 3, task: str = "regression") -> None:
        super().__init__()
        if task not in {"regression", "classification"}:
            raise ValueError("task must be'regression' or 'classification'")
        self.task = task
        self.num_features = num_features
        self.depth = depth

        # Build the core network
        self.net = self._build_network(num_features, depth, task)
        # Metadata that mirrors the quantum interface
        self.encoding: List[int] = list(range(num_features))
        self.weight_sizes: List[int] = self._compute_weight_sizes()
        self.output_dim = 1 if task == "regression" else 2

    def _build_network(self, num_features: int, depth: int, task: str) -> nn.Sequential:
        layers: List[nn.Module] = []

        # Optional initial embedding – keeps the number of hidden units equal to input size
        layers.append(nn.Linear(num_features, num_features))
        layers.append(nn.Tanh())

        for _ in range(depth):
            layers.append(nn.Linear(num_features, num_features))
            if task == "classification":
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Tanh())

        # Final head
        layers.append(nn.Linear(num_features, self.output_dim))
        return nn.Sequential(*layers)

    def _compute_weight_sizes(self) -> List[int]:
        sizes: List[int] = []
        for module in self.net:
            if isinstance(module, nn.Linear):
                sizes.append(module.weight.numel() + module.bias.numel())
        return sizes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_encoding(self) -> List[int]:
        """Return the list of feature indices that are directly fed into the network."""
        return self.encoding

    def get_weight_sizes(self) -> List[int]:
        """Return a list of the parameter counts for each linear layer."""
        return self.weight_sizes

    def get_observables(self) -> List[int]:
        """
        Return a placeholder list of observables that matches the quantum interface.
        For a classical model the observables are simply indices of the output nodes.
        """
        return list(range(self.output_dim))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_features={self.num_features}, depth={self.depth}, task={self.task})"

__all__ = ["EstimatorQNNGen042"]
