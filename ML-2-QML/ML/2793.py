"""Hybrid classical implementation of a fully‑connected quantum‑style layer and classifier.

The class mirrors the structure of the quantum reference while providing a completely classical forward pass.
It can be used as a drop‑in replacement for the original FCL factory, but with an added classifier head.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import numpy as np


class HybridQuantumClassifier(nn.Module):
    """
    Classical feed‑forward network that emulates a quantum fully‑connected layer
    followed by a classifier. The architecture is inspired by the ``FCL`` and
    ``QuantumClassifierModel`` seeds.
    """

    def __init__(self, num_features: int = 1, depth: int = 1, device: str = "cpu") -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.device = device

        # Fully‑connected “quantum” layer (linear + tanh)
        self.fc_layer = nn.Linear(num_features, 1, bias=True)

        # Classifier network (depth layers of Linear → ReLU)
        layers = []
        in_dim = 1  # output of fc_layer
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        # Head: 2‑class logits
        layers.append(nn.Linear(in_dim, 2))
        self.classifier = nn.Sequential(*layers)

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the fully‑connected layer and the classifier.
        """
        x = x.to(self.device)
        out = torch.tanh(self.fc_layer(x))
        logits = self.classifier(out)
        return logits

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Mimic the quantum ``run`` interface: treat ``thetas`` as a batch of
        input features, compute the forward pass, and return the mean
        probability of class 1.
        """
        values = torch.tensor(list(thetas), dtype=torch.float32, device=self.device).view(-1, self.num_features)
        logits = self.forward(values)
        probs = torch.softmax(logits, dim=1)[:, 1]
        expectation = probs.mean().detach().cpu().numpy()
        return np.array([expectation])


def FCL() -> HybridQuantumClassifier:
    """
    Factory compatible with the original ``FCL`` seed.  Returns an instance
    of the hybrid classical classifier with default parameters.
    """
    return HybridQuantumClassifier()


__all__ = ["HybridQuantumClassifier", "FCL"]
