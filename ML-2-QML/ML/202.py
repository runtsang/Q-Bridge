"""Enhanced classical classifier with feature scaling, regularisation and modular head."""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

import torch
import torch.nn as nn
import sklearn.preprocessing as sp

class QuantumClassifierModel(nn.Module):
    """A modular feed‑forward network mirroring the quantum design but with optional
    dropout, L2 regularisation and a built‑in StandardScaler for feature
    normalisation. The class also exposes a static helper that returns the
    network together with metadata used by the quantum counterpart."""
    def __init__(self,
                 num_features: int,
                 depth: int,
                 hidden_sizes: Optional[List[int]] = None,
                 dropout: float = 0.0,
                 weight_decay: float = 0.0,
                 device: str = "cpu") -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.hidden_sizes = hidden_sizes or [num_features] * depth
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.device = device

        layers: List[nn.Module] = []
        in_dim = num_features
        for hidden in self.hidden_sizes:
            linear = nn.Linear(in_dim, hidden)
            nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
            layers.append(linear)
            layers.append(nn.ReLU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(p=self.dropout))
            in_dim = hidden
        head = nn.Linear(in_dim, 2)
        nn.init.xavier_uniform_(head.weight)
        layers.append(head)

        self.network = nn.Sequential(*layers).to(self.device)
        self.scaler = sp.StandardScaler()

    def fit_scaler(self, X: torch.Tensor) -> None:
        """Fit the internal StandardScaler to the provided data."""
        self.scaler.fit(X.cpu().numpy())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional scaling."""
        x = self.scaler.transform(x.cpu().numpy())
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        return self.network(x)

    @staticmethod
    def build_classifier_circuit(num_features: int,
                                 depth: int,
                                 hidden_sizes: Optional[List[int]] = None,
                                 dropout: float = 0.0,
                                 weight_decay: float = 0.0,
                                 device: str = "cpu") -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """Construct a feed‑forward classifier and expose metadata that mirrors the quantum variant."""
        model = QuantumClassifierModel(num_features, depth, hidden_sizes, dropout, weight_decay, device)
        encoding = list(range(num_features))
        weight_sizes = [p.numel() for p in model.network.parameters()]
        observables = list(range(2))
        return model.network, encoding, weight_sizes, observables

__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
