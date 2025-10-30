"""Enhanced classical counterpart to the hybrid quantum binary classifier.

This module extends the original architecture by adding batch‑normalisation,
residual connections, and a two‑output head that mirrors the quantum
expectation layer.  The hybrid head is a simple learnable sigmoid that
provides a probabilistic output compatible with the QML version.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Hybrid(nn.Module):
    """Dense head that replaces the quantum circuit.

    The head maps the scalar feature to a single logit and applies a
    learnable shift before the sigmoid.  The shift is optimised
    jointly with the rest of the network and allows the model to
    emulate the bias behaviour of the quantum expectation layer.
    """
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = nn.Parameter(torch.tensor(shift, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = self.linear(inputs)
        probs = torch.sigmoid(logits + self.shift)
        return probs


class QCNet(nn.Module):
    """Convolutional binary classifier with a hybrid‑style head.

    The network mirrors the QML architecture but replaces the quantum
    expectation layer with the classical `Hybrid` module.  Two output
    logits are produced by `fc3` and the hybrid head returns the
    probability of class 1.  The final output is a probability
    vector of shape ``(batch, 2)``.
    """
    def __init__(self) -> None:
        super().__init__()
        # Feature extractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(15)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Classifier
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)   # two logits for the hybrid head

        # Hybrid head
        self.hybrid = Hybrid(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.bn1(self.conv1(inputs)))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)          # shape (batch, 2)
        prob_class1 = self.hybrid(x[:, 0:1])  # use first logit as input to hybrid
        return torch.cat((prob_class1, 1 - prob_class1), dim=-1)


__all__ = ["Hybrid", "QCNet"]
