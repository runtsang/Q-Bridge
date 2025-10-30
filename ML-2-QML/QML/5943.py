"""Hybrid self‑attention regression – quantum implementation.

The quantum module mirrors the classical architecture: a
self‑attention style variational circuit extracts features from the
encoded state, followed by a measurement‑based regression head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create states of the form cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>.
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
    Dataset providing quantum states and regression targets.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class QuantumSelfAttention(tq.QuantumModule):
    """
    Variational self‑attention block implemented with rotation and entangling gates.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        # Parameterised rotations
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        # Entangling layer
        self.crx = tq.CRX(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        # Apply a layer of random rotations
        for w in range(self.num_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)
            self.rz(qdev, wires=w)
        # Entangle neighbours
        for i in range(self.num_wires - 1):
            self.crx(qdev, wires=[i, i + 1])


class HybridSelfAttentionRegression(tq.QuantumModule):
    """
    Quantum hybrid model: encoder → self‑attention circuit → measurement → regression head.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        # Encode classical features into amplitudes
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        # Self‑attention block
        self.attention = QuantumSelfAttention(num_wires)
        # Measurement to classical bits
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Regression head
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : Tensor of shape (batch, num_wires)
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        # Encode
        self.encoder(qdev, state_batch)
        # Self‑attention variational layer
        self.attention(qdev)
        # Measure
        features = self.measure(qdev)
        # Regression
        return self.head(features).squeeze(-1)


__all__ = ["HybridSelfAttentionRegression", "RegressionDataset", "generate_superposition_data"]
