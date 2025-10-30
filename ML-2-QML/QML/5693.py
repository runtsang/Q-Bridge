"""Quantum regression dataset and model with noise and multi‑output support."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from typing import Tuple

def generate_superposition_data(
    num_wires: int,
    samples: int,
    output_dim: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>.
    Generates multi‑output labels as random linear combinations of sin(2theta) and cos(phi).
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

    # Base labels: sin(2theta) * cos(phi) and sin(2theta) * sin(phi)
    base = np.stack([np.sin(2 * thetas) * np.cos(phis), np.sin(2 * thetas) * np.sin(phis)], axis=1)
    # Random linear combination to produce output_dim
    W = np.random.randn(output_dim, 2).astype(np.float32)
    labels = base @ W.T
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset that returns a dictionary with "states" and "target".
    """
    def __init__(self, samples: int, num_wires: int, output_dim: int = 1):
        self.states, self.labels = generate_superposition_data(num_wires, samples, output_dim)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> dict:
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(tq.QuantumModule):
    """
    Quantum regression model with optional Gaussian noise to emulate decoherence.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int, output_dim: int = 1, noise_level: float = 0.0):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, output_dim)
        self.noise_level = noise_level

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        if self.noise_level > 0.0:
            noise = torch.randn_like(features) * self.noise_level
            features = features + noise
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
