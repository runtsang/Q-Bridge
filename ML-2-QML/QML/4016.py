"""Quantum regression module – variational circuit with controlled‑CNOT entanglement.

This module mirrors the classical baseline but replaces the linear head
with a parametrised quantum circuit.  It relies on torchquantum for
device management, random layers and gate primitives.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple

def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cosθ|0…0⟩ + e^{iφ} sinθ|1…1⟩."""
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
    """Dataset that returns quantum states and labels."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(tq.QuantumModule):
    """Variational quantum regression network."""

    class QLayer(tq.QuantumModule):
        """Layer that applies a random circuit, single‑qubit rotations
        and a controlled‑CNOT entanglement pattern.
        """

        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            # Pre‑compute CNOT pattern: a ring of CNOTs (last qubit controls first)
            self.cnot_pattern: list[tuple[int, int]] = [
                (i, i + 1) for i in range(n_wires - 1)
            ] + [(n_wires - 1, 0)]

        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)
            # Controlled‑CNOT ring
            for ctrl, tgt in self.cnot_pattern:
                tqf.cnot(qdev, wires=[ctrl, tgt])
            return qdev

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder that maps a real vector to a superposition state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
