"""Enhanced classical classifier with residual layers and dropout."""
from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumClassifier(nn.Module):
    """
    Feed‑forward classifier with optional residual connections, batch‑normalisation
    and dropout.  The architecture is parameterised by ``depth`` and ``hidden_dim``.
    """
    def __init__(self, num_features: int, depth: int,
                 hidden_dim: int | None = None,
                 dropout: float = 0.0,
                 device: str | torch.device = "cpu"):
        super().__init__()
        hidden_dim = hidden_dim or num_features
        self.layers: nn.ModuleList = nn.ModuleList()
        in_dim = num_features
        for _ in range(depth):
            seq = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
            )
            self.layers.append(seq)
            in_dim = hidden_dim
        self.head = nn.Linear(in_dim, 2)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connections."""
        out = x
        for layer in self.layers:
            residual = out
            out = layer(out)
            out = out + residual  # residual addition
        logits = self.head(out)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def weight_sizes(self) -> List[int]:
        """Return the number of parameters per trainable module."""
        return [p.numel() for p in self.parameters()]

def build_classifier_circuit(num_features: int, depth: int,
                             hidden_dim: int | None = None,
                             dropout: float = 0.0) -> Tuple[QuantumClassifier,
                                                           Iterable[int],
                                                           List[int],
                                                           List[int]]:
    """
    Construct a classifier instance and expose metadata mirroring the quantum
    helper interface.  ``observables`` are simply the indices of the output
    classes in this classical setting.
    """
    model = QuantumClassifier(num_features, depth, hidden_dim=hidden_dim,
                              dropout=dropout)
    encoding = list(range(num_features))
    weight_sizes = model.weight_sizes()
    observables = [0, 1]  # two-class output
    return model, encoding, weight_sizes, observables

__all__ = ["QuantumClassifier", "build_classifier_circuit"]
