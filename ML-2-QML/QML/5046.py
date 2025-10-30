"""Hybrid regression model with a variational quantum circuit.

The quantum version closely follows the original ``QuantumRegression`` seed but
adds a feature‑normalising layer and a batch‑norm head to match the classical
counterpart.  The encoder maps each input dimension to a separate qubit
using a Ry rotation; a random layer injects entanglement, and the measurement
provides a real‑valued feature vector for the linear head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate data from a superposition of |0…0> and |1…1>.

    Parameters
    ----------
    num_wires : int
        Number of qubits (input dimensions).
    samples : int
        Number of data points to generate.

    Returns
    -------
    states : np.ndarray
        Complex state vectors of shape (samples, 2**num_wires).
    labels : np.ndarray
        Regression targets.
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


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset wrapper around ``generate_superposition_data``."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class HybridRegressionModel(tq.QuantumModule):
    """Quantum regression model that parallels the classical counterpart.

    The architecture uses a GeneralEncoder to map each input feature to a
    separate qubit via a Ry rotation, a RandomLayer for entanglement, and a
    global Pauli‑Z measurement.  The resulting feature vector is passed through
    a linear head followed by BatchNorm1d for output scaling.
    """

    class QLayer(tq.QuantumModule):
        """Variational layer with a random circuit and parametrised rotations."""

        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder: each feature is encoded into a separate qubit with Ry
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)
        self.norm = nn.BatchNorm1d(num_wires)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode the classical data into the quantum state
        self.encoder(qdev, state_batch)
        # Apply the variational layer
        self.q_layer(qdev)
        # Extract expectation values
        features = self.measure(qdev)
        # Classical linear head with batch‑norm
        out = self.head(features)
        return self.norm(out).squeeze(-1)


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
