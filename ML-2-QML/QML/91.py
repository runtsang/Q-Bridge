"""Quantum regression model using Pennylane with a feature‑map and variational ansatz."""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data in the computational basis.

    Parameters
    ----------
    num_wires : int
        Number of qubits used to encode the input state.
    samples : int
        Number of data points.
    """
    # Basis states |0..0> and |1..1>
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
    Dataset wrapper for the synthetic quantum regression data.
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


class RegressionModel(nn.Module):
    """
    Quantum‑enhanced regression model.

    Parameters
    ----------
    num_wires : int
        Number of qubits.
    feature_map_depth : int, optional
        Number of layers in the feature‑map. Default is 2.
    ansatz_layers : int, optional
        Number of layers in the variational ansatz. Default is 2.
    device : str, optional
        Pennylane device name. Default is 'default.qubit'.
    """

    def __init__(
        self,
        num_wires: int,
        feature_map_depth: int = 2,
        ansatz_layers: int = 2,
        device: str = "default.qubit",
    ):
        super().__init__()
        self.num_wires = num_wires
        self.dev = qml.device(device, wires=num_wires)

        def circuit(state, params):
            # Feature map: angle embedding followed by entangling layers
            qml.templates.embeddings.AngleEmbedding(params, wires=range(num_wires))
            for _ in range(feature_map_depth):
                qml.templates.layers.StronglyEntanglingLayers(params, wires=range(num_wires))

            # Variational ansatz
            qml.templates.layers.StronglyEntanglingLayers(params, wires=range(num_wires))

            # Measure expectation values of PauliZ on all wires
            return [qml.expval(qml.PauliZ(w)) for w in range(num_wires)]

        # Number of parameters: (feature_map_depth + ansatz_layers) * num_wires * 3
        num_params = (feature_map_depth + ansatz_layers) * num_wires * 3
        self.qnode = qml.QNode(circuit, self.dev, interface="torch")

        # Classical head to map quantum measurements to a scalar output
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of quantum states.

        Parameters
        ----------
        state_batch : torch.Tensor
            Tensor of shape (batch_size, 2**num_wires, 2) representing complex amplitudes.
        """
        # Convert complex tensor to real angles for embedding
        # Here we simply take the phase of the amplitude as a placeholder
        phases = torch.angle(state_batch).view(state_batch.shape[0], -1)
        # Pad or truncate to match expected input dimension
        if phases.shape[-1] > self.num_wires:
            phases = phases[:, : self.num_wires]
        elif phases.shape[-1] < self.num_wires:
            pad = torch.zeros(*phases.shape[:-1], self.num_wires - phases.shape[-1], device=phases.device)
            phases = torch.cat([phases, pad], dim=-1)

        # Broadcast parameters for all samples
        params = torch.randn((state_batch.shape[0], (2 * self.num_wires * 3)))  # placeholder
        # Execute QNode in batch mode
        qmeas = self.qnode(phases, params)
        qmeas = torch.stack(qmeas, dim=1)  # (batch, num_wires)
        return self.head(qmeas).squeeze(-1)

    def quantum_kernel_matrix(self, data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel matrix between two batches of states.

        Parameters
        ----------
        data1 : torch.Tensor
            Tensor of shape (n1, 2**num_wires, 2).
        data2 : torch.Tensor
            Tensor of shape (n2, 2**num_wires, 2).
        """
        n1, n2 = data1.shape[0], data2.shape[0]
        kernel = torch.zeros((n1, n2), device=data1.device)
        for i in range(n1):
            for j in range(n2):
                # Inner product of two quantum states
                overlap = torch.dot(data1[i].flatten(), data2[j].flatten().conj()).abs()
                kernel[i, j] = overlap
        return kernel


__all__ = ["RegressionModel", "RegressionDataset", "generate_superposition_data"]
