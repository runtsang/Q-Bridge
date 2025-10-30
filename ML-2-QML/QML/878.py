"""Quantum regression model built with Pennylane.

The architecture encodes the two‑dimensional angle vector using RX/RY
rotations, applies a variational circuit with CNOT couplings, and
measures Pauli‑Z expectation values.  A classical linear head maps the
features to a scalar prediction.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic quantum‑style data.

    The function returns a pair of arrays:
        - ``states``: shape (samples, num_features) of angles (theta, phi).
        - ``labels``: regression targets ``sin(2*theta) * cos(phi)``.
    """
    x = np.random.uniform(0, 2 * np.pi, size=(samples, num_features)).astype(np.float32)
    thetas = x[:, 0]
    phis = x[:, 1]
    y = np.sin(2 * thetas) * np.cos(phis)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset yielding angle vectors and regression targets."""

    def __init__(self, samples: int, num_features: int = 2):
        self.states, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuantumRegressionModel(nn.Module):
    """
    Quantum regression model using Pennylane.

    Parameters
    ----------
    num_wires : int
        Number of qubits; must be >= 2 because we encode two angles.
    num_layers : int
        Number of variational layers.
    device : str
        Pennylane backend (default.qubit, default.mixed, etc.).
    """

    def __init__(
        self,
        num_wires: int = 2,
        num_layers: int = 3,
        device: str = "default.qubit",
    ):
        super().__init__()
        self.num_wires = num_wires
        self.num_layers = num_layers
        self.dev = qml.device(device, wires=num_wires)

        # Variational parameters: shape (num_layers, num_wires, 3)
        self.weights = nn.Parameter(
            torch.randn(num_layers, num_wires, 3, dtype=torch.float32)
        )

        self.head = nn.Linear(num_wires, 1)

        # Compile the QNode once
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Variational circuit returning expectation values of Pauli‑Z."""
        # Encode the two angles using RX and RY rotations
        qml.RX(x[0], wires=0)
        qml.RY(x[1], wires=1)

        # Variational layers
        for layer in range(self.num_layers):
            for wire in range(self.num_wires):
                qml.Rot(
                    weights[layer, wire, 0],
                    weights[layer, wire, 1],
                    weights[layer, wire, 2],
                    wires=wire,
                )
            for i in range(self.num_wires - 1):
                qml.CNOT(wires=[i, i + 1])

        # Measure expectation of Pauli‑Z on each wire
        return [qml.expval(qml.PauliZ(w)) for w in range(self.num_wires)]

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass over a batch of input states.

        Parameters
        ----------
        state_batch : torch.Tensor
            Shape (batch, num_features) where the first two elements are the
            angles used for encoding.  Remaining elements are ignored.

        Returns
        -------
        torch.Tensor
            Shape (batch,) regression predictions.
        """
        # Pennylane QNode does not support batched execution out of the box,
        # so we loop over the batch.  For large batches consider using
        # ``batch_size`` on the device or a custom batched QNode.
        batch_results = []
        for i in range(state_batch.shape[0]):
            out = self.qnode(state_batch[i], self.weights)
            batch_results.append(out)
        features = torch.stack(batch_results, dim=0)
        return self.head(features).squeeze(-1)


__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data"]
