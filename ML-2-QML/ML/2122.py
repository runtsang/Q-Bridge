from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List

__all__ = ["build_classifier_circuit", "QuantumClassifierModel"]


class QuantumClassifierModel(nn.Module):
    """
    Classical feed‑forward classifier that mimics the quantum helper interface.
    The network depth, dropout probability and loss type can be tuned for experimentation.
    """

    def __init__(self, num_features: int, depth: int, dropout: float = 0.0,
                 loss_type: str = "cross_entropy"):
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.dropout = dropout
        self.loss_type = loss_type

        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = num_features
        self.head = nn.Linear(in_dim, 2)
        layers.append(self.head)
        self.network = nn.Sequential(*layers)

        # Metadata to resemble the quantum variant
        self.encoding = list(range(num_features))
        self.weight_sizes = [p.numel() for p in self.parameters()]
        self.observables = [0, 1]  # simple class indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "cross_entropy":
            return F.cross_entropy(logits, labels)
        elif self.loss_type == "mse":
            return F.mse_loss(F.softmax(logits, dim=1),
                               F.one_hot(labels, num_classes=2).float())
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                   optimizer: torch.optim.Optimizer) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss


def build_classifier_circuit(num_features: int, depth: int,
                             dropout: float = 0.0,
                             loss_type: str = "cross_entropy") -> Tuple[nn.Module,
                             Iterable[int], Iterable[int], List[int]]:
    """
    Construct a feed‑forward classifier and return the network along with
    metadata that mirrors the quantum helper's interface.
    """
    model = QuantumClassifierModel(num_features, depth, dropout, loss_type)
    return model.network, model.encoding, model.weight_sizes, model.observables
