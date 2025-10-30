"""
QuantumRegressionHybrid: Classical‑quantum regression model.
Author: Auto‑Generated
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Data generation (classical)
# --------------------------------------------------------------------------- #
def generate_superposition_data(
    num_features: int,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic regression dataset.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input vectors.
    samples : int
        Number of samples to generate.

    Returns
    -------
    x : np.ndarray
        Feature matrix of shape (samples, num_features).
    y : np.ndarray
        Target vector of shape (samples,).
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class RegressionDataset(Dataset):
    """
    PyTorch Dataset that returns feature tensors and their scalar targets.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Hybrid model
# --------------------------------------------------------------------------- #
# Import the quantum refinement block.  It is defined in the accompanying
# quantum module `QuantumRegressionHybrid` (see qml_code below).  The import
# is wrapped in a try/except so that the module can be used independently
# if the quantum block is not available.
try:
    from.quantum_regression import QuantumRegressionHybrid as QuantumBlock
except Exception:
    QuantumBlock = None

class QuantumRegressionHybrid(nn.Module):
    """
    Classic‑quantum hybrid regression model.

    The architecture consists of:
        * A classical MLP backbone that projects the input to a hidden space.
        * A quantum refinement block that receives the hidden representation,
          encodes it, runs a parameterised variational circuit, measures
          expectation values, and maps them to a scalar.
        * A final linear head that produces the regression output.
    """
    def __init__(
        self,
        num_features: int,
        num_wires: int,
        hidden_sizes: tuple[int,...] = (64, 32),
    ):
        """
        Parameters
        ----------
        num_features : int
            Dimensionality of the input vectors.
        num_wires : int
            Number of qubits used in the quantum refinement block.
        hidden_sizes : tuple[int,...]
            Sequence of hidden layer sizes for the classical backbone.
        """
        super().__init__()

        # Classical backbone
        layers = []
        in_dim = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        # Ensure that the final hidden dimension matches the number of qubits
        if in_dim!= num_wires:
            raise ValueError(
                f"Final hidden dimension ({in_dim}) must equal num_wires ({num_wires})."
            )

        # Quantum refinement block
        if QuantumBlock is None:
            raise ImportError(
                "Quantum block not available. Import the quantum module "
                "`QuantumRegressionHybrid` from `quantum_regression.py`."
            )
        self.quantum_block = QuantumBlock(num_wires)

        # Final head
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, num_features).

        Returns
        -------
        torch.Tensor
            Regression output of shape (batch,).
        """
        # Classical feature extraction
        features = self.backbone(x)

        # Quantum refinement
        q_features = self.quantum_block(features)

        # Final linear head
        out = self.head(q_features)
        return out.squeeze(-1)

__all__ = ["QuantumRegressionHybrid", "RegressionDataset", "generate_superposition_data"]
