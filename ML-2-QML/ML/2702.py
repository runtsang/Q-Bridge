"""Hybrid classical neural network that serves as a feature extractor for the quantum estimator.

Features:
- Two-stage feed‑forward architecture: an encoding block (linear+ReLU) followed by a variational block that outputs parameters for the quantum circuit.
- Supports both regression and classification by providing separate heads.
- The output of the variational block can be directly used as the weight parameters for the quantum circuit.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple

def EstimatorQNN() -> nn.Module:
    """Return a hybrid classical neural network.

    The network produces:
    - `q_params`: a vector of variational parameters for the quantum circuit.
    - `regression`: a scalar for regression tasks.
    - `classification`: logits for binary classification.

    The architecture mirrors the classical counterpart of the original EstimatorQNN
    and the QuantumClassifierModel, but adds an explicit variational block that
    outputs the required number of parameters for a depth‑d quantum ansatz.
    """
    class HybridNN(nn.Module):
        def __init__(self, input_dim: int = 2, hidden_dim: int = 8, depth: int = 2, num_qubits: int = 1) -> None:
            super().__init__()
            # Encoding block
            self.encoding = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
            )
            # Variational block: outputs parameters for the quantum circuit
            self.variational = nn.Sequential(
                nn.Linear(hidden_dim, num_qubits * depth),
                nn.Tanh(),
            )
            # Heads
            self.reg_head = nn.Linear(num_qubits * depth, 1)
            self.cls_head = nn.Linear(num_qubits * depth, 2)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Return (q_params, regression, classification)."""
            encoded = self.encoding(x)
            q_params = self.variational(encoded)
            regression = self.reg_head(q_params)
            classification = self.cls_head(q_params)
            return q_params, regression, classification

    return HybridNN()
