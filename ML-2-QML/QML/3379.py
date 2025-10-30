"""Hybrid regression model with a quantum convolutional circuit."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0…0⟩ + e^{iφ} sin(theta)|1…1⟩."""
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
    """Quantum variational circuit with convolution‑style layers for regression."""
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            # Convolutional block: RX, RY, CZ on consecutive pairs
            self.conv_ops = nn.ModuleList()
            for wire in range(0, num_wires, 2):
                self.conv_ops.append(tq.RX(has_params=True, trainable=True))
                self.conv_ops.append(tq.RY(has_params=True, trainable=True))
                self.conv_ops.append(tq.CZ())

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for i, wire in enumerate(range(0, self.n_wires, 2)):
                self.conv_ops[3 * i](qdev, wires=wire)
                self.conv_ops[3 * i + 1](qdev, wires=wire + 1)
                self.conv_ops[3 * i + 2](qdev, wires=[wire, wire + 1])

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Feature map: encode input amplitudes into qubit states
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        # After pooling we keep the first wire of each pair
        self.head = nn.Linear(num_wires // 2, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)  # shape (bsz, n_wires)
        pooled = features[:, ::2]  # keep first wire of each pair
        return self.head(pooled).squeeze(-1)


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
