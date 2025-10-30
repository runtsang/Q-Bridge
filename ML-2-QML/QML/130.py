"""Quantum regression module with entangling layers and dual‑observable measurement.

Key extensions:
* `GeneralEncoder` now uses a custom Ry‑rotation per wire.
* Adds a stack of parameterised CX (controlled‑NOT) layers to entangle the qubits.
* Measures both Pauli‑Z and Pauli‑X expectations, concatenating them as features.
* The classical head is a deeper MLP that outputs both mean and log‑variance.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

__all__ = ["QuantumRegression", "RegressionDataset", "generate_superposition_data"]


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Create states of the form cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>."""
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
    """Dataset that returns complex state vectors and scalar targets."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class EntangleLayer(tq.QuantumModule):
    """Parameterised CX‑stack that entangles all wires."""

    def __init__(self, num_wires: int, depth: int = 2):
        super().__init__()
        self.depth = depth
        self.cx_layers = nn.ModuleList(
            [tq.CX(control=0, target=i) for i in range(1, num_wires)]
        ) * depth

    def forward(self, qdev: tq.QuantumDevice) -> None:
        for cx in self.cx_layers:
            cx(qdev)


class QuantumRegression(tq.QuantumModule):
    """Quantum regression model with dual‑observable measurement."""

    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder: single‑qubit Ry rotations
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.entangle = EntangleLayer(num_wires)
        # Dual‑observable measurement
        self.measure_z = tq.MeasureAll(tq.PauliZ)
        self.measure_x = tq.MeasureAll(tq.PauliX)
        # Classical head: mean & log‑variance
        self.head = nn.Sequential(
            nn.Linear(2 * num_wires, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(32, 1)
        self.logvar_head = nn.Linear(32, 1)

    def forward(self, state_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        self.entangle(qdev)
        # Concatenate Z and X expectation values
        features = torch.cat([self.measure_z(qdev), self.measure_x(qdev)], dim=-1)
        hidden = self.head(features)
        mean = self.mean_head(hidden).squeeze(-1)
        logvar = self.logvar_head(hidden).squeeze(-1)
        return mean, logvar

    def loss(self, mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Negative log‑likelihood for Gaussian output."""
        var = torch.exp(logvar)
        return 0.5 * torch.mean((target - mean) ** 2 / var + logvar)

    def train_step(self, optimizer: torch.optim.Optimizer, batch: dict):
        self.train()
        optimizer.zero_grad()
        mean, logvar = self.forward(batch["states"])
        loss = self.loss(mean, logvar, batch["target"])
        loss.backward()
        optimizer.step()
        return loss.item()
