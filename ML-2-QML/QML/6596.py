"""Hybrid quantum‑classical regression with an entangling variational circuit and multi‑observable measurement."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


def generate_superposition_data(num_wires: int, samples: int, noise: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate complex amplitude states and target labels."""
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
    if noise > 0.0:
        labels += noise * np.random.randn(samples).astype(np.float32)
    return states, labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset yielding quantum state tensors and regression targets."""

    def __init__(self, samples: int, num_wires: int, noise: float = 0.0):
        self.states, self.labels = generate_superposition_data(num_wires, samples, noise)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QModel(tq.QuantumModule):
    """Hybrid model: amplitude‑encoded input → variational circuit → multi‑observable measurement → classical head."""

    class QLayer(tq.QuantumModule):
        """Variational layer with entanglement and parameterized rotations."""

        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.cnot = tq.CNOT

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)
                self.rz(qdev, wires=wire)
            # Entangle adjacent qubits
            for wire in range(self.n_wires - 1):
                self.cnot(qdev, wires=[wire, wire + 1])

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Amplitude encoding via a general encoder
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        # Measure expectation values of PauliZ on all wires
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical head: 2‑layer MLP
        self.head = nn.Sequential(
            nn.Linear(num_wires, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)  # shape: (bsz, n_wires)
        return self.head(features).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
