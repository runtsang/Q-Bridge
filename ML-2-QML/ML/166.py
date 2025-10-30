"""Hybrid classical-quantum fraud detection model.

This module defines a PyTorch model that extracts features from
transaction data, passes them to a quantum circuit implemented
in Pennylane, then refines the result with a second linear head.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

# Import the quantum layer from the QML module
# The QML module must be available on the Python path
from fraud_detection_qml import quantum_layer


@dataclass
class FraudHybridParams:
    """Parameters for the hybrid model."""
    feature_dim: int          # Dimensionality of the input transaction vector
    hidden_dim: int           # Size of the hidden layer in the classical extractor
    quantum_param_dim: int    # Number of parameters in the variational circuit


class FraudDetectionHybrid(nn.Module):
    """Hybrid classical-quantum fraud detection model."""

    def __init__(self, params: FraudHybridParams) -> None:
        super().__init__()
        # Classical feature extractor
        self.extractor = nn.Sequential(
            nn.Linear(params.feature_dim, params.hidden_dim),
            nn.ReLU(),
            nn.Linear(params.hidden_dim, params.hidden_dim),
            nn.ReLU(),
        )
        # Quantum circuit parameters (learnable)
        init_q_params = torch.randn(params.quantum_param_dim, dtype=torch.float32)
        self.q_params = nn.Parameter(init_q_params)
        # Final classifier
        self.classifier = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract classical features
        h = self.extractor(x)
        # Compute quantum output; detach h to avoid back‑prop through the extractor
        q_out = quantum_layer(h.detach(), self.q_params)
        # Pass quantum output through a final linear layer
        out = self.classifier(q_out)
        return torch.sigmoid(out)
