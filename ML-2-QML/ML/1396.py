"""QuantumHybridClassifier – classical‑only counterpart with enriched head and optional residual.

This module mirrors the architecture of the original hybrid model but replaces the
quantum expectation head with a lightweight classical head.  The head can be
configured to output a binary or multi‑class probability distribution and
optionally add a residual connection from the input.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QuantumHybridClassifier"]

class ClassicalHybridHead(nn.Module):
    """
    Dense head that processes the output of the last fully‑connected layer.
    Parameters
    ----------
    in_features : int
        Size of the input feature vector.
    out_features : int, default=1
        Number of logits to produce.
    use_residual : bool, default=False
        If True, add a residual connection from the input.
    activation : str, default='sigmoid'
        Activation function to apply to the logits.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int = 1,
                 use_residual: bool = False,
                 activation: str = "sigmoid") -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.residual = nn.Linear(in_features, out_features, bias=False) if use_residual else None
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x)
        if self.residual is not None:
            logits += self.residual(x)
        return getattr(torch, self.activation)(logits)

class QuantumHybridClassifier(nn.Module):
    """
    Classical‑only binary classifier that mimics the architecture of the
    hybrid quantum network.  The final layer is a configurable dense head
    that can optionally add a residual connection.
    """
    def __init__(self,
                 use_residual: bool = False,
                 head_out_features: int = 1,
                 activation: str = "sigmoid") -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully‑connected head
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Classical hybrid head
        self.head = ClassicalHybridHead(
            in_features=self.fc3.out_features,
            out_features=head_out_features,
            use_residual=use_residual,
            activation=activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional backbone
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)

        # Flatten and fully‑connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Classical head
        probs = self.head(x)

        # For binary classification return a 2‑dim distribution
        if probs.ndim == 1:
            probs = probs.unsqueeze(-1)
        return torch.cat((probs, 1 - probs), dim=-1)
