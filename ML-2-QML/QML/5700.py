"""Quantum regression model using PennyLane.

This module builds a hybrid variational circuit that encodes input states
via amplitude encoding, applies a trainable ansatz, and outputs a vector
of Pauli‑Z expectation values that are fed into a classical linear head.
The dataset mirrors the classical version but returns complex tensors
suitable for quantum simulation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from torch.utils.data import Dataset
from typing import Tuple


def generate_superposition_data(
    num_wires: int,
    samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate states of the form
    cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>.
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Dataset returning complex amplitude‑encoded states.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuantumRegressionModel(nn.Module):
    """
    Hybrid variational circuit for regression.

    Parameters
    ----------
    num_wires : int
        Number of qubits in the circuit.
    layers : int, optional
        Number of variational layers. Default: 3.
    params_per_layer : int, optional
        Number of rotation parameters per layer. Default: 3.
    """
    def __init__(
        self,
        num_wires: int,
        layers: int = 3,
        params_per_layer: int = 3,
    ) -> None:
        super().__init__()
        self.num_wires = num_wires

        # Device with automatic differentiation via Torch
        self.device = qml.device("default.qubit", wires=num_wires)

        # Weight matrix for the ansatz
        weight_shape = (layers, params_per_layer, num_wires)
        self.weights = nn.Parameter(torch.randn(weight_shape, dtype=torch.float32))

        # Classical linear head
        self.head = nn.Linear(num_wires, 1)

        # Register the QNode
        @qml.qnode(self.device, interface="torch")
        def circuit(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            # Amplitude encoding
            qml.QubitStateVector(x, wires=range(num_wires))
            # Variational layers
            for l in range(layers):
                for p in range(params_per_layer):
                    for w_idx in range(num_wires):
                        qml.RY(w[l, p, w_idx], wires=w_idx)
                # Entanglement across all wires
                for j in range(num_wires):
                    qml.CNOT(wires=[j, (j + 1) % num_wires])
            # Expectation values of Pauli‑Z on each wire
            return [qml.expval(qml.PauliZ(i)) for i in range(num_wires)]

        self.circuit = circuit

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor of shape (batch, 2**num_wires)
            Complex amplitude‑encoded states.
        """
        bsz = state_batch.shape[0]
        # Run the circuit in a loop; PennyLane does not yet support batch
        # evaluation with a Torch interface for arbitrary shapes.
        features = torch.stack(
            [self.circuit(state_batch[i], self.weights) for i in range(bsz)],
            dim=0,
        )
        return self.head(features).squeeze(-1)


__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data"]
