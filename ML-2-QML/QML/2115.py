"""Quantum regression module using PennyLane with amplitude encoding and a hybrid variational circuit.

The implementation extends the seed by:
* Amplitude‑encoding of classical features into a quantum state,
* A flexible variational ansatz composed of parameterised rotation layers
  and entangling CNOTs, allowing the circuit to learn non‑linear mappings.
* A measurement step that returns the expectation value of Pauli‑Z on all
  qubits, which is then fed into a classical linear head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from torch.utils.data import Dataset


def generate_superposition_data(
    num_wires: int,
    samples: int,
    noise_std: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset that can be amplitude‑encoded into a quantum state.

    Parameters
    ----------
    num_wires:
        Number of qubits used for encoding.
    samples:
        Number of data points.
    noise_std:
        Gaussian noise added to the target.
    """
    dim = 2 ** num_wires
    # Random complex vectors uniformly distributed on the hypersphere
    states = (
        np.random.normal(size=(samples, dim))
        + 1j * np.random.normal(size=(samples, dim))
    )
    states = states / np.linalg.norm(states, axis=1, keepdims=True)

    # Simple target function of the state's phases
    angles = np.angle(states).sum(axis=1)
    labels = np.sin(angles) + 0.1 * np.cos(2 * angles)

    labels += np.random.normal(0.0, noise_std, size=labels.shape)

    return states.astype(np.complex64), labels.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset providing amplitude‑encoded states and targets."""

    def __init__(self, samples: int, num_wires: int, **kwargs):
        self.states, self.labels = generate_superposition_data(num_wires, samples, **kwargs)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


def _circuit(params: torch.Tensor, wires: list[int], num_layers: int):
    num_wires = len(wires)
    for layer in range(num_layers):
        for i in range(num_wires):
            qml.RY(params[layer, i], wires=wires[i])
            qml.RZ(params[layer, i + num_wires], wires=wires[i])
        # Circular CNOT entanglement
        for i in range(num_wires):
            qml.CNOT(wires[i], wires[(i + 1) % num_wires])


class QModel(nn.Module):
    """
    Hybrid quantum‑classical regression model.

    The model consists of:
    * An amplitude‑encoding layer that maps a classical vector to a quantum state.
    * A variational ansatz defined by ``_circuit`` with learnable parameters.
    * A measurement of ⟨Z⟩ on each qubit followed by a linear head.
    """

    def __init__(self, num_wires: int, num_layers: int = 3):
        super().__init__()
        self.num_wires = num_wires
        self.num_layers = num_layers

        # Parameter matrix: one rotation per wire per layer, for RY and RZ
        self.params = nn.Parameter(torch.randn(num_layers, 2 * num_wires, dtype=torch.float32))

        # PennyLane device
        self.dev = qml.device("default.qubit", wires=num_wires, shots=0)

        # Classical head
        self.head = nn.Linear(num_wires, 1)

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(state: torch.Tensor, params: torch.Tensor):
            # Prepare the quantum state via amplitude encoding
            qml.QubitStateVector(state, wires=range(self.num_wires))
            # Variational layers
            _circuit(params, list(range(self.num_wires)), self.num_layers)
            # Return expectation values of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_wires)]

        return circuit

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch
            Batch of amplitude‑encoded feature vectors of shape (B, 2**num_wires).
        """
        batch_size = state_batch.shape[0]
        circuit = self._build_qnode()
        # Compute expectation values for every sample in the batch
        features = torch.stack(
            [circuit(state_batch[i], self.params) for i in range(batch_size)],
            dim=0,
        )
        return self.head(features).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
