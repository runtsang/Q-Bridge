"""Extended classical classifier mirroring the quantum interface.

Features
--------
* Configurable hidden size, depth, activation, dropout.
* Returns weight sizes and encoding indices.
* Provides method to compute probabilities.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumClassifierModel(nn.Module):
    """
    A PyTorch feed‑forward classifier that mirrors the API of the quantum
    counterpart.  It exposes the same helper attributes used by the QML
    implementation: `encoding`, `weight_sizes`, and `observables`.
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 32,
        depth: int = 3,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        """
        Parameters
        ----------
        num_features: int
            Number of input features.
        hidden_size: int, optional
            Width of each hidden layer.
        depth: int, optional
            Number of hidden layers.
        activation: str, optional
            Activation function ('relu', 'tanh', 'gelu').
        dropout: float, optional
            Dropout probability applied after each hidden layer.
        """
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.depth = depth

        act = self._get_activation(activation)
        layers: List[nn.Module] = []

        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(act)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_size

        # final head
        layers.append(nn.Linear(in_dim, 2))
        self.network = nn.Sequential(*layers)

        # metadata to mimic the quantum interface
        self.encoding: List[int] = list(range(num_features))
        self.weight_sizes: List[int] = [
            p.numel() for p in self.network.parameters()
        ]
        self.observables: List[int] = [0, 1]  # dummy observable indices

    def _get_activation(self, name: str) -> nn.Module:
        """Return activation module given name."""
        if name.lower() == "relu":
            return nn.ReLU()
        if name.lower() == "tanh":
            return nn.Tanh()
        if name.lower() == "gelu":
            return nn.GELU()
        raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class indices."""
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=-1)


__all__ = ["QuantumClassifierModel"]
