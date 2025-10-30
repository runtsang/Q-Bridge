"""Hybrid classical estimator that mirrors the EstimatorQNN example but adds
clipping and scaling inspired by the FraudDetection seed.

The network outputs a single learnable weight that drives a 1‑qubit
quantum circuit.  The classic part is a 2‑layer feed‑forward network
with ReLU activations and a final linear head.  The weight is
clipped to a user‑defined bound to keep the quantum parameters
stable during training.  The quantum expectation value is obtained
by calling the quantum estimator defined in the companion QML
module (EstimatorQNNGenQuantum).  This module is fully
importable and can be used standalone for purely classical
experiments or combined with the quantum module for hybrid
training.

Usage
-----
>>> from EstimatorQNN__gen195 import EstimatorQNNGen
>>> model = EstimatorQNNGen()
>>> x = torch.randn(4, 2)        # batch of 4 samples, 2 features
>>> out = model(x)               # forward pass returns a tensor of shape (4, 1)
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple

# Import the quantum estimator from the companion QML module.
# The import is optional; if the QML module is missing the
# model will still run but the quantum part will be a no‑op.
try:
    from EstimatorQNN__gen195_qml import EstimatorQNNGenQuantum
except Exception:  # pragma: no cover
    EstimatorQNNGenQuantum = None  # type: ignore[assignment]


def _clip(value: float, bound: float) -> float:
    """Clip a scalar to the range [-bound, bound]."""
    return max(-bound, min(bound, value))


class EstimatorQNNGen(nn.Module):
    """
    Classical + quantum hybrid estimator.

    Parameters
    ----------
    clip_bound : float, optional
        Upper bound for clipping the weight parameter that drives the quantum
        circuit.  A value of 0 disables clipping.
    """

    def __init__(self, clip_bound: float = 5.0) -> None:
        super().__init__()
        self.clip_bound = clip_bound

        # Classical sub‑network
        self.classical = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

        # Weight that will be fed to the quantum circuit.
        self.weight = nn.Parameter(torch.tensor(0.0))

        # Optional quantum estimator
        self.quantum = EstimatorQNNGenQuantum() if EstimatorQNNGenQuantum else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid estimator.

        The classical network produces a scalar weight that is clipped
        and then passed to the quantum estimator.  The final output
        is the expectation value of the quantum circuit, optionally
        concatenated with the classical prediction.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (*, 2).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (*, 1).
        """
        # Classical prediction (used only for regularisation if desired)
        class_pred = self.classical(inputs)

        # Clip the weight before sending it to the quantum circuit
        clipped_weight = _clip(self.weight.item(), self.clip_bound)

        if self.quantum:
            # Quantum expectation value
            q_output = self.quantum(clipped_weight)
            # Combine classical and quantum outputs (simple sum here)
            output = class_pred + q_output
        else:
            # Fallback: return classical prediction only
            output = class_pred

        return output


__all__ = ["EstimatorQNNGen"]
