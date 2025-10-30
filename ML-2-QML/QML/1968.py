"""Quantum regression model with entanglement and dual‑basis measurement.

This version enriches the original seed by:
* optional entanglement via CNOT gates,
* a trainable RZ rotation per wire,
* simultaneous measurement of Pauli‑Z and Pauli‑X,
* a two‑layer classical head that fuses the quantum features.
"""

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


def generate_superposition_data(num_wires: int, samples: int,
                                noise_std: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data with optional label noise."""
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
    if noise_std > 0.0:
        labels += np.random.normal(scale=noise_std, size=labels.shape).astype(np.float32)
    return states, labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Quantum dataset wrapper."""
    def __init__(self, samples: int, num_wires: int, noise_std: float = 0.0):
        self.states, self.labels = generate_superposition_data(num_wires, samples, noise_std)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QModel(tq.QuantumModule):
    """Hybrid quantum‑classical regression network."""
    class QLayer(tq.QuantumModule):
        """Parameterised quantum layer with optional entanglement."""
        def __init__(self, num_wires: int, entangle: bool = True, n_random: int = 30):
            super().__init__()
            self.n_wires = num_wires
            self.entangle = entangle
            self.random_layer = tq.RandomLayer(n_ops=n_random, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)
                self.rz(qdev, wires=wire)
            if self.entangle:
                for wire in range(self.n_wires):
                    tq.CNOT(wires=(wire, (wire + 1) % self.n_wires))(qdev)

    def __init__(self, num_wires: int, head_hidden: int = 16, entangle: bool = True):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires, entangle=entangle)
        # Measure both Z and X to capture real and imaginary components
        self.measure_z = tq.MeasureAll(tq.PauliZ)
        self.measure_x = tq.MeasureAll(tq.PauliX)
        # Classical head: two‑layer MLP
        self.head = nn.Sequential(
            nn.Linear(2 * num_wires, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        z_features = self.measure_z(qdev)
        x_features = self.measure_x(qdev)
        features = torch.cat([z_features, x_features], dim=1)
        return self.head(features).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
