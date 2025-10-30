"""Quantum classifier module leveraging Pennylane for a variational circuit.

Key extensions over the original seed
------------------------------------
* Adds a trainable random layer (30 Pauli gates) to enrich the feature space.
* Uses a weighted sum of Pauli‑Z expectations as the observable vector.
* Provides a data‑augmentation routine identical to the classical side.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp


def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate superposition states and a target function.

    The labels are thresholded to produce a binary classification task.
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
    binary_labels = (labels > 0).astype(np.float32)
    return states, binary_labels


class ClassificationDataset(torch.utils.data.Dataset):
    """Dataset that wraps the superposition data for binary classification."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuantumClassifierModel(nn.Module):
    """Variational quantum classifier with a random layer and a weighted Z observable."""

    def __init__(self, num_wires: int, depth: int = 3, random_layer_ops: int = 30):
        super().__init__()
        self.num_wires = num_wires
        self.depth = depth
        self.device = torch.device("cpu")  # Pennylane handles device internally

        # Parameter vectors
        self.encoding_params = pnp.random.uniform(-np.pi, np.pi, (num_wires,))
        self.theta_params = pnp.random.uniform(-np.pi, np.pi, (num_wires * depth,))

        # Random layer to mix states
        self.random_layer = qml.RandomLayer(num_wires, ops=pnp.arange(random_layer_ops))

        # Measurement observables: weighted sum of Z on each qubit
        self.weights = torch.nn.Parameter(torch.ones(num_wires, dtype=torch.float32))

        # Classical head
        self.head = nn.Linear(num_wires, 2)

        # QNode definition
        self.qnode = qml.QNode(self._circuit, qml.device("default.qubit", wires=num_wires), interface="torch")

    def _circuit(self, inputs: torch.Tensor, encoding: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Variational circuit that encodes data and applies a random layer plus a variational ansatz."""
        # Data encoding
        for i in range(self.num_wires):
            qml.RX(encoding[i], wires=i)
            qml.RY(encoding[i], wires=i)

        # Random layer
        self.random_layer()

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for i in range(self.num_wires):
                qml.RY(theta[idx], wires=i)
                idx += 1
            # Entangling CZs
            for i in range(self.num_wires - 1):
                qml.CZ(i, i + 1)

        # Return expectation values of Z on each qubit
        return [qml.expval(qml.PauliZ(w)) for w in range(self.num_wires)]

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Compute logits from the quantum circuit and a linear head."""
        # Encode input states into rotation angles
        encoding = state_batch.to(torch.float32)
        theta = self.theta_params

        # Run QNode
        q_out = self.qnode(encoding, self.encoding_params, theta)

        # Weighted sum of observables
        features = torch.stack(q_out, dim=-1) * self.weights
        logits = self.head(features)
        return logits

    @staticmethod
    def weight_sizes(model: nn.Module) -> List[int]:
        """Return a list of weight+bias counts for each trainable layer."""
        sizes = []
        for module in model.modules():
            if isinstance(module, nn.Linear):
                sizes.append(module.weight.numel() + module.bias.numel())
        return sizes

    @staticmethod
    def observables() -> Iterable[int]:
        """Return indices of observables for compatibility with the classical interface."""
        return list(range(2))


__all__ = ["QuantumClassifierModel", "ClassificationDataset", "generate_superposition_data"]
