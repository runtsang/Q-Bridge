"""Enhanced classical classifier mirroring the quantum interface.

The class accepts a flexible network topology and dropout, and
provides metadata used by the quantum side: encoding indices,
weight sizes and observable indices.  It also exposes a simple
forward pass and a convenience method to retrieve a parameter
dict for checkpointing.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumClassifierModel:
    """Neural‑network counterpart of the quantum classifier.

    Parameters
    ----------
    num_features : int
        Number of input features (also the number of qubits in the quantum model).
    hidden_sizes : Iterable[int] | None
        Sizes of hidden layers.  If ``None`` a single hidden layer equal to
        ``num_features`` is constructed.  The sequence is interpreted as a
        chain ``num_features -> hidden_sizes[0] ->... -> hidden_sizes[-1]``.
    dropout : float
        Dropout probability applied after each ReLU.  Defaults to ``0.0``.
    """

    def __init__(
        self,
        num_features: int,
        hidden_sizes: Iterable[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        self.num_features = num_features
        self.hidden_sizes = list(hidden_sizes) if hidden_sizes else [num_features]
        self.dropout = dropout
        self._build_network()

    # ------------------------------------------------------------------ #
    #  Network construction helpers
    # ------------------------------------------------------------------ #
    def _build_network(self) -> None:
        layers: List[nn.Module] = []
        last_dim = self.num_features
        for hidden in self.hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(nn.ReLU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            last_dim = hidden
        layers.append(nn.Linear(last_dim, 2))  # binary classification
        self.network = nn.Sequential(*layers)

        # Metadata: weight sizes for each linear layer and observable indices
        self.weight_sizes: List[int] = []
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                self.weight_sizes.append(m.weight.numel() + m.bias.numel())
        self.encoding: List[int] = list(range(self.num_features))
        self.observables: List[int] = list(range(2))

    # ------------------------------------------------------------------ #
    #  Forward pass
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict:
        """Return the underlying network parameters."""
        return self.network.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Load parameters into the network."""
        self.network.load_state_dict(state_dict)

    def get_encoding(self) -> List[int]:
        """Return the list of feature indices used for encoding."""
        return self.encoding

    def get_weight_sizes(self) -> List[int]:
        """Return the sizes of all trainable weight tensors."""
        return self.weight_sizes

    def get_observables(self) -> List[int]:
        """Return the observable indices (for compatibility with the QML side)."""
        return self.observables

__all__ = ["QuantumClassifierModel"]
