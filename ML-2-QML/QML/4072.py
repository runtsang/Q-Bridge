"""Hybrid quantum‑classical regression model combining an encoder, a variational quantum layer,
a quantum fully‑connected layer, and a classical linear head.

The structure mirrors the classical HybridRegression but replaces the classical
encoder and sampler with quantum counterparts.  The quantum fully‑connected layer
uses a single Ry gate on a qubit and measures the Z expectation value, emulating
the behaviour of the classical `FCL` layer.  The overall architecture
demonstrates how classical and quantum components can be fused into a single
differentiable module.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset

__all__ = ["generate_superposition_data", "RegressionDataset", "HybridRegression"]


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Same data generation as the original seed but adapted to the quantum version.
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
    return states, labels


class RegressionDataset(Dataset):
    """
    Dataset that returns complex quantum states and scalar labels.
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


class QuantumFullyConnectedLayer(tq.QuantumModule):
    """
    A single‑qubit quantum circuit that applies a parameterized Ry gate
    and measures the Z expectation value.  This mimics the behaviour of
    the classical `FCL` layer but in a quantum circuit.
    """
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.r_y = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.r_y(qdev, wires=0)
        return tq.MeasureAll(tq.PauliZ)(qdev)[:, 0:1]


class HybridRegression(tq.QuantumModule):
    """
    Quantum‑classical regression model.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self._build_variational_layer(num_wires)
        self.q_fc = QuantumFullyConnectedLayer(num_wires)
        self.head = nn.Linear(num_wires + 1, 1)

    def _build_variational_layer(self, n_wires: int) -> tq.QuantumModule:
        class QLayer(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
                self.rx = tq.RX(has_params=True, trainable=True)
                self.ry = tq.RY(has_params=True, trainable=True)

            def forward(self, qdev: tq.QuantumDevice):
                self.random_layer(qdev)
                for wire in range(self.n_wires):
                    self.rx(qdev, wires=wire)
                    self.ry(qdev, wires=wire)

        return QLayer(n_wires)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        q_fc_out = self.q_fc(qdev)
        features = tq.MeasureAll(tq.PauliZ)(qdev)
        combined = torch.cat([features, q_fc_out], dim=-1)
        return self.head(combined).squeeze(-1)
