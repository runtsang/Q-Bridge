"""Classical hybrid‑quantum classifier with residual feature extractor.

The class ``HybridQuantumClassifier`` implements a deep residual
feed‑forward network that maps raw inputs to a vector of qubit angles.
It can be paired with a quantum head (e.g. ``QuantumExpectationHead``)
via ``set_quantum_head``.  The output is a probability distribution
over two classes, obtained by a sigmoid applied to the quantum output
plus an optional bias shift.

This construction blends the feed‑forward factory from
``QuantumClassifierModel.py`` and the hybrid head from
``ClassicalQuantumBinaryClassification.py`` while adding a residual
structure for better gradient flow.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class HybridQuantumClassifier(nn.Module):
    """
    Classical feature extractor with an optional quantum head.

    Parameters
    ----------
    in_features : int
        Dimensionality of the input feature vector.
    n_qubits : int
        Number of qubits the quantum head operates on.
    hidden_dim : int, default=64
        Width of the hidden layers in the residual extractor.
    depth : int, default=3
        Number of residual blocks.
    shift : float, default=0.0
        Bias added before the sigmoid activation.
    device : str or torch.device, default="cpu"
        Target device for the model.
    """
    def __init__(self,
                 in_features: int,
                 n_qubits: int,
                 hidden_dim: int = 64,
                 depth: int = 3,
                 shift: float = 0.0,
                 device: str | torch.device = "cpu"):
        super().__init__()
        self.n_qubits = n_qubits
        self.shift = shift
        self.device = device

        # Residual feed‑forward extractor
        layers = []
        dim = in_features
        for _ in range(depth):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        self.extractor = nn.Sequential(*layers)

        # Map to qubit angles
        self.to_qubits = nn.Linear(hidden_dim, n_qubits)

        # Placeholder for quantum head
        self.quantum_head: Optional[nn.Module] = None

    def set_quantum_head(self, head: nn.Module) -> None:
        """Attach a quantum head that implements a forward returning a scalar."""
        self.quantum_head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_features).

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, 2) containing class probabilities.
        """
        x = self.extractor(x)
        qubit_angles = self.to_qubits(x)
        if self.quantum_head is None:
            raise RuntimeError("Quantum head not attached")
        q_out = self.quantum_head(qubit_angles)
        probs = torch.sigmoid(q_out + self.shift)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridQuantumClassifier"]
