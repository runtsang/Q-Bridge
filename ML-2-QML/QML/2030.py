"""Quantum regression model using Pennylane.

The module implements :class:`QuantumRegression` that wraps a variational
ansatz built with Pennylane.  The circuit uses an ``AngleEmbedding`` of the
input features followed by a ``StronglyEntanglingLayers`` ansatz.  The
expectation values of Pauli‑Z on each qubit are fed into a classical linear
head to produce the regression output.

The dataset generator mirrors the classical version but produces samples
in the form of angles suitable for the quantum encoder.

Author: GPT‑OSS‑20B
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pennylane as qml
import pennylane.numpy as jnp


def generate_quantum_dataset(num_qubits: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset of angles and labels for training the quantum circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input features.
    samples : int
        Number of samples to generate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``features`` of shape ``(samples, num_qubits)`` with values in
        ``[0, 2π]`` and ``labels`` of shape ``(samples,)``.
    """
    angles = np.random.uniform(0.0, 2 * np.pi, size=(samples, num_qubits)).astype(np.float32)
    angles_sum = angles.sum(axis=1)
    labels = np.sin(angles_sum) + 0.1 * np.cos(2 * angles_sum)
    return angles, labels.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Torch ``Dataset`` for the quantum regression task.
    """

    def __init__(self, samples: int, num_qubits: int):
        self.features, self.labels = generate_quantum_dataset(num_qubits, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuantumRegression(nn.Module):
    """
    Hybrid classical‑quantum regression model.

    The quantum part is a Pennylane variational circuit that accepts a batch
    of input angles.  The device is configured with ``shots=None`` for
    exact gradients.  The output of the circuit is a vector of Pauli‑Z
    expectation values, which serves as features for a linear read‑out.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input features.
    num_layers : int, optional
        Number of layers in the ``StronglyEntanglingLayers`` ansatz.
    """

    def __init__(self, num_qubits: int, num_layers: int = 2, device: str = "cpu"):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.device = device

        # PennyLane device for exact simulation
        self.dev = qml.device("default.qubit", wires=num_qubits, shots=None, backend="default.qubit")

        # Trainable weights for the ansatz
        self.weights = nn.Parameter(
            torch.randn(num_layers, num_qubits, 3, dtype=torch.float32)
        )

        # Classical head
        self.head = nn.Linear(num_qubits, 1)

        # Build the QNode
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            qml.templates.AngleEmbedding(inputs, wires=range(num_qubits))
            qml.templates.StronglyEntanglingLayers(weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self.circuit = circuit

    def forward(self, batch_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass over a batch of input angles.

        Parameters
        ----------
        batch_inputs : torch.Tensor
            Shape ``(batch, num_qubits)``.

        Returns
        -------
        torch.Tensor
            Regression predictions of shape ``(batch,)``.
        """
        # Pennylane can handle batches via the ``batch_size`` argument in the
        # device, but to keep the example simple we loop over the batch.
        expvals = []
        for x in batch_inputs:
            expvals.append(self.circuit(x, self.weights))
        expvals = torch.stack(expvals)  # shape (batch, num_qubits)
        return self.head(expvals).squeeze(-1)


__all__ = ["QuantumRegression", "RegressionDataset", "generate_quantum_dataset"]
