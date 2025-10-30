"""Quantum regression module with hybrid classical‑quantum feature extractor."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset

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

class RegressionDataset(Dataset):
    """Dataset yielding quantum states and regression targets."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class ResidualBlock(nn.Module):
    """A residual dense block for the classical branch."""
    def __init__(self, dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class VariationalAnsatz(tq.QuantumModule):
    """Structured variational ansatz: RX, RZ, CNOT layers."""
    def __init__(self, n_wires: int, layers: int = 3):
        super().__init__()
        self.n_wires = n_wires
        self.layers = layers
        for i in range(layers):
            self.add_module(f"rx_{i}", tq.RX(has_params=True, trainable=True))
            self.add_module(f"rz_{i}", tq.RZ(has_params=True, trainable=True))
            self.add_module(f"cnot_{i}", tq.CNOT(has_params=False, trainable=False))

    def forward(self, qdev: tq.QuantumDevice):
        for i in range(self.layers):
            self[f"rx_{i}"](qdev)
            self[f"rz_{i}"](qdev)
            self[f"cnot_{i}"](qdev)

class QuantumRegression(tq.QuantumModule):
    """Hybrid quantum‑classical regression model."""
    def __init__(self, num_wires: int, residual_depth: int = 1):
        super().__init__()
        self.n_wires = num_wires
        # Parameter‑free encoder: simple Ry rotations on each wire
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        # Variational ansatz
        self.variational = VariationalAnsatz(num_wires)
        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical branch
        self.classical_branch = nn.Sequential(
            nn.Linear(num_wires, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.classical_residuals = nn.ModuleList([ResidualBlock(16) for _ in range(residual_depth)])
        # Fusion head
        self.fusion_head = nn.Linear(16 + num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode
        self.encoder(qdev, state_batch)
        # Variational
        self.variational(qdev)
        # Measure
        quantum_features = self.measure(qdev)
        # Classical
        classical_features = self.classical_branch(state_batch)
        for res in self.classical_residuals:
            classical_features = res(classical_features)
        # Fuse
        fused = torch.cat([quantum_features, classical_features], dim=-1)
        return self.fusion_head(fused).squeeze(-1)

__all__ = ["QuantumRegression", "RegressionDataset", "generate_superposition_data"]
