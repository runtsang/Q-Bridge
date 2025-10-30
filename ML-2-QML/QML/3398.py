"""Hybrid quantum‑classical regression model with a quantum self‑attention layer."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic quantum states and labels."""
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
    """Dataset wrapping the synthetic quantum states."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumSelfAttention(tq.QuantumModule):
    """Parameterised quantum self‑attention block."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        # Trainable rotation angles
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        # Trainable entangling gate
        self.crx = tq.CRX(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        # Apply independent rotations on each wire
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)
            self.rz(qdev, wires=w)
        # Entangle adjacent wires with controlled‑RX
        for w in range(self.n_wires - 1):
            self.crx(qdev, control=w, target=w + 1)

class HybridRegressor(tq.QuantumModule):
    """Quantum‑classical regression pipeline:
    * Encode classical state
    * Quantum self‑attention
    * Random feature layer
    * Measurement → linear head."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder mapping classical vector → state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        # Quantum self‑attention block
        self.attention = QuantumSelfAttention(num_wires)
        # Feature‑extractor
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical head
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode input
        self.encoder(qdev, state_batch)
        # Quantum self‑attention
        self.attention(qdev)
        # Random layer for richer feature mapping
        self.random_layer(qdev)
        # Measure
        features = self.measure(qdev)
        # Classical prediction
        return self.head(features).squeeze(-1)

__all__ = ["HybridRegressor", "RegressionDataset", "generate_superposition_data"]
