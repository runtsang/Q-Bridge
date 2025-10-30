"""Quantum regression model that applies a quanvolution‑style kernel to 2‑D feature maps."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate superposition states with 4‑wire patches for quanvolution."""
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
    """Dataset exposing complex quantum states and scalar targets."""
    def __init__(self, samples: int, num_wires: int):
        super().__init__()
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(tq.QuantumModule):
    """Quantum regression model that applies a quanvolution filter followed by a variational layer."""
    class QuanvolutionLayer(tq.QuantumModule):
        """Apply a random two‑qubit quantum kernel to 2×2 patches."""
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "ry", "wires": [0]},
                    {"input_idx": [1], "func": "ry", "wires": [1]},
                    {"input_idx": [2], "func": "ry", "wires": [2]},
                    {"input_idx": [3], "func": "ry", "wires": [3]},
                ]
            )
            self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, qdev: tq.QuantumDevice, batch: torch.Tensor) -> torch.Tensor:
            # batch shape: (bsz, 2^num_wires)
            patches = batch.view(-1, self.n_wires)
            self.encoder(qdev, patches)
            self.q_layer(qdev)
            return self.measure(qdev)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.qfilter = self.QuanvolutionLayer(num_wires)
        self.q_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Apply quanvolution filter
        features = self.qfilter(qdev, state_batch)
        # Variational transformation
        self.q_layer(qdev)
        self.rx(qdev)
        self.ry(qdev)
        variational = self.qfilter.measure(qdev)
        return self.head(variational).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
