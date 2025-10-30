"""Hybrid quantum regression model combining quantum FCLayer, quanvolution, and transformer blocks."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
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
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumFCLayer(tq.QuantumModule):
    """Quantum fully‑connected layer (FCL) that returns the mean Pauli‑Z expectation."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.random_layer(qdev)
        return self.measure(qdev).mean(dim=1, keepdim=True)

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Quantum quanvolution filter that processes the full state as a 28×28 image."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.random_layer(qdev)
        return self.measure(qdev)

class QuantumTransformerBlock(tq.QuantumModule):
    """Simple quantum transformer block that applies two QLayers to simulate attention and feed‑forward."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=10, wires=list(range(n_wires)))
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.random_layer(qdev)
            return self.measure(qdev)

    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.attn = self.QLayer(n_wires)
        self.ffn = self.QLayer(n_wires)
        self.linear = nn.Linear(n_wires, n_wires)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        attn_out = self.attn(qdev)
        ffn_out = self.ffn(qdev)
        combined = attn_out + ffn_out
        return self.linear(combined)

class HybridRegressionModel(tq.QuantumModule):
    """Hybrid quantum regression model combining quantum FCLayer, quanvolution, and transformer."""
    def __init__(self, num_wires: int = 10, n_qubits_transformer: int = 8):
        super().__init__()
        self.num_wires = num_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [idx], "func": "ry", "wires": [idx]}
                for idx in range(num_wires)
            ]
        )
        self.qfclayer = QuantumFCLayer(num_wires)
        self.qquanvolution = QuantumQuanvolutionFilter()
        self.transformer = QuantumTransformerBlock(n_qubits_transformer)
        self.head = nn.Linear(2 * num_wires + 1, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        # FCLayer
        qdev_fcl = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev_fcl, state_batch)
        fcl_out = self.qfclayer(qdev_fcl)  # (batch, 1)
        # Quanvolution
        qdev_quanv = tq.QuantumDevice(n_wires=self.qquanvolution.n_wires, bsz=bsz, device=state_batch.device)
        self.qquanvolution.encoder(qdev_quanv, state_batch)
        quanv_features = self.qquanvolution(qdev_quanv)  # (batch, n_wires)
        # Transformer
        qdev_transformer = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev_transformer, state_batch)
        transformer_features = self.transformer(qdev_transformer)  # (batch, n_wires)
        # Combine
        combined = torch.cat([transformer_features, fcl_out, quanv_features], dim=-1)
        return self.head(combined).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
