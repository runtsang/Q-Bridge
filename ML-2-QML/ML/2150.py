"""Classical classifier with probabilistic outputs and a modular training interface.

The class mirrors the quantum helper's interface while adding:
* A configurable encoder depth.
* Optional dropout for regularisation.
* Utility methods for loss computation, accuracy, and F1‑score.
* A static ``build_classifier_circuit`` factory that returns the same network
  structure used by the quantum version, facilitating direct comparison.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumClassifierModel(nn.Module):
    """
    A PyTorch feed‑forward network that emulates the quantum helper's API.

    Parameters
    ----------
    num_features : int
        Number of input features (``num_qubits`` in the quantum version).
    depth : int
        Number of hidden layers in the encoder.
    dropout : float, optional
        Dropout probability applied after each hidden layer (default: 0.0).
    """

    def __init__(self, num_features: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.dropout = dropout

        layers: list[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = num_features
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits."""
        x = self.encoder(x)
        return self.head(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities using softmax."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices."""
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)

    def compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Cross‑entropy loss (unnormalised)."""
        return F.cross_entropy(logits, y)

    def accuracy(self, logits: torch.Tensor, y: torch.Tensor) -> float:
        """Compute accuracy over a batch."""
        preds = self.predict(logits)
        return (preds == y).float().mean().item()

    def f1_score(self, logits: torch.Tensor, y: torch.Tensor) -> float:
        """Compute macro‑averaged F1‑score over a batch."""
        from sklearn.metrics import f1_score

        preds = self.predict(logits).cpu().numpy()
        target = y.cpu().numpy()
        return f1_score(target, preds, average="macro")

    @staticmethod
    def build_classifier_circuit(
        num_features: int, depth: int
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
        """
        Factory that recreates the architecture used by the quantum version
        for direct comparison.

        Returns
        -------
        network : nn.Module
            The constructed feed‑forward network.
        encoding : Iterable[int]
            Dummy encoding list that matches the quantum encoding indices.
        weight_sizes : Iterable[int]
            List of parameter counts per layer.
        observables : list[int]
            Placeholder observables for the quantum interface.
        """
        layers: list[nn.Module] = []
        in_dim = num_features
        weight_sizes = []
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.extend([linear, nn.ReLU()])
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())
        network = nn.Sequential(*layers)
        encoding = list(range(num_features))
        observables = list(range(2))
        return network, encoding, weight_sizes, observables


__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
