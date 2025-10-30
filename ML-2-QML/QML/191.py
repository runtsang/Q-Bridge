"""Quantum regression model with a multi‑layer parameterized ansatz and hybrid head.

The quantum architecture extends the seed by introducing entangling gates,
dual‑observable feature extraction (Pauli‑Z and Pauli‑X), and a small
classical neural network head.  The public API (`QModel`, `RegressionDataset`,
`generate_superposition_data`) remains unchanged, enabling drop‑in
replacement while providing a richer quantum feature space.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
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
    return states, labels

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset for the synthetic quantum regression problem."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumRegressionAnsatz(tq.QuantumModule):
    """Parameterised ansatz consisting of random layers, rotations, and entanglement."""
    def __init__(self, n_wires: int, n_layers: int = 3):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.random_layers = nn.ModuleList(
            [tq.RandomLayer(n_ops=20, wires=list(range(n_wires))) for _ in range(n_layers)]
        )
        self.rotation = nn.ModuleList(
            [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
        )

    def forward(self, qdev: tq.QuantumDevice):
        for layer in self.random_layers:
            layer(qdev)
        # Parameterised rotations on each wire
        for wire, rot in enumerate(self.rotation):
            rot(qdev, wires=wire)
        # Entanglement via CNOT on adjacent wires
        for i in range(self.n_wires - 1):
            tq.CNOT(has_params=False, trainable=False)(qdev, wires=[i, i + 1])

class QModel(tq.QuantumModule):
    """
    Quantum regression model with a hybrid classical head.

    Parameters
    ----------
    num_wires : int
        Number of qubits in the device.
    n_layers : int, optional
        Depth of the parameterised ansatz.  Defaults to 3.
    """
    def __init__(self, num_wires: int, n_layers: int = 3):
        super().__init__()
        self.n_wires = num_wires
        # Encoder maps classical amplitudes to a quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        # Parameterised ansatz
        self.ansatz = QuantumRegressionAnsatz(num_wires, n_layers)
        # Dual‑observable measurement
        self.measure_z = tq.MeasureAll(tq.PauliZ)
        self.measure_x = tq.MeasureAll(tq.PauliX)
        # Classical head: maps concatenated expectation values to a scalar
        self.head = nn.Sequential(
            nn.Linear(num_wires * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.ansatz(qdev)
        z_features = self.measure_z(qdev)
        x_features = self.measure_x(qdev)
        features = torch.cat([z_features, x_features], dim=1)
        return self.head(features).squeeze(-1)

    def predict(self, state_batch: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        """Convenience wrapper for inference."""
        self.eval()
        with torch.no_grad():
            if device is not None:
                state_batch = state_batch.to(device)
            return self.forward(state_batch)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
