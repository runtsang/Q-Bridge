"""Quantum regression model extended with entangling layers and dual‑basis measurement."""

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
    """
    PyTorch Dataset wrapping the synthetic superposition data for quantum training.
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


class RegressionModel(tq.QuantumModule):
    """Variational quantum circuit with entanglement and dual‑basis measurement."""

    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            # Random parametric layer
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            # Parameterised rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            # Entanglement gate
            self.cnot = tq.CNOT()

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)
            # Entangle adjacent wires
            for i in range(self.n_wires - 1):
                self.cnot(qdev, control=i, target=i + 1)

    def __init__(self, num_wires: int, num_layers: int = 3):
        super().__init__()
        self.n_wires = num_wires
        self.num_layers = num_layers

        # Feature encoding (same as seed for compatibility)
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        # Build variational layers
        self.layers = nn.ModuleList([self.QLayer(num_wires) for _ in range(num_layers)])
        # Dual‑basis measurement
        self.measure_z = tq.MeasureAll(tq.PauliZ)
        self.measure_x = tq.MeasureAll(tq.PauliX)
        # Classical head mapping concatenated features to output
        self.head = nn.Linear(2 * num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        for layer in self.layers:
            layer(qdev)
        z_features = self.measure_z(qdev)
        x_features = self.measure_x(qdev)
        features = torch.cat([z_features, x_features], dim=1)
        return self.head(features).squeeze(-1)


__all__ = ["RegressionModel", "RegressionDataset", "generate_superposition_data"]
