"""Classical classifier factory mirroring the quantum helper interface with extended features."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumClassifierModel(nn.Module):
    """
    A flexible feed‑forward neural network that mimics the interface of the quantum
    classifier.  The architecture is parametrised by ``num_features``, ``depth`` and
    ``hidden_dim``.  Optional dropout and batch‑normalisation layers can be enabled
    to improve generalisation.  The class exposes ``train_step`` and ``predict`` so
    it can be used as a drop‑in replacement for the original ``build_classifier_circuit``.
    """
    def __init__(self, num_features: int, depth: int = 2, hidden_dim: int | None = None,
                 dropout: float = 0.0, batchnorm: bool = False) -> None:
        super().__init__()
        hidden_dim = hidden_dim or num_features
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2))
        self.network = nn.Sequential(*layers)

        # Metadata mirroring the quantum implementation
        self.encoding = list(range(num_features))
        self.weight_sizes = [p.numel() for p in self.parameters()]
        self.observables = [0, 1]  # dummy observables

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def train_step(self, x: torch.Tensor, y: torch.Tensor,
                   optimizer: torch.optim.Optimizer,
                   loss_fn: nn.Module = nn.CrossEntropyLoss()) -> torch.Tensor:
        self.train()
        optimizer.zero_grad()
        logits = self.forward(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        return loss

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            return (probs[:, 1] > threshold).long()


def build_classifier_circuit(num_features: int, depth: int = 2) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Backwards‑compatible helper that returns an instance of ``QuantumClassifierModel`` and
    the metadata expected by legacy code.
    """
    model = QuantumClassifierModel(num_features, depth)
    return model, model.encoding, model.weight_sizes, model.observables


__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
