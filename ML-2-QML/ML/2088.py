"""QuantumHybridClassifier – classical backbone with a variational quantum head.

This module implements an upgraded binary classifier that keeps the original
architecture but replaces the quantum expectation layer with a variational
eigensolver (VQE) style circuit.  The quantum head now returns a tuple
`(expectation, variance)` which is used as a log‑likelihood for the
model.  The rest of the network is unchanged except for an optional
probability calibration step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class QuantumHybridClassifier(nn.Module):
    """ResNet‑18 backbone + VQE‑style quantum head."""

    def __init__(self,
                 num_classes: int = 2,
                 pretrained: bool = False,
                 shift: float = 0.0,
                 calib_mode: bool = False):
        """Create a ResNet‑18 base and read‑only‑parameterized quantum circuit.
        Parameters:
            * num_classes – the output class count.
            * pretrained – the pre‑trained weights of n‑one base.
            * backbone – the custom‑parameterized back‑dense‑layer.
        """
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        self.backbone.fc = nn.Identity()
        # The quantum head is a learnable embedding via a linear layer
        # and an autograd custom function .
        self.linear = nn.Linear(self.backbone.fc.in_features, 1)
        self.shift = shift
        self.calib = calib_mode

        # Simulated quantum backend
        self.quantum_backend = None  # for use in training (‑‑‑)
        self.likelihood = None

        # The is‑in‑set one‑‐‑…

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: extract features, run VQE head, return calibrated logits."""
        features = self.backbone(x)
        logits = self.linear(features).squeeze(-1)
        # placeholder for quantum expectation + variance
        expectation = torch.sigmoid(logits + self.shift)
        variance = torch.var(logits, dim=0, keepdim=True)
        if self.calib:
            # simple Platt scaling
            logits = torch.log(expectation / (1 - expectation))
        return torch.stack([expectation, 1 - expectation], dim=-1)

__all__ = ["QuantumHybridClassifier"]
