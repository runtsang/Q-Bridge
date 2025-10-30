"""Quantum regression module with parameter‑sharing rotations and auto‑differentiable encoder."""

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
    return states, labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset wrapper for the synthetic quantum regression data."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class ParameterSharedLayer(tq.QuantumModule):
    """Rotations with a single shared parameter per gate type across all wires."""

    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.rz_params = nn.Parameter(torch.randn(1))
        self.rx_params = nn.Parameter(torch.randn(1))
        self.ry_params = nn.Parameter(torch.randn(1))
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice):
        for wire in range(self.num_wires):
            self.rz(qdev, wires=wire, params=self.rz_params)
            self.rx(qdev, wires=wire, params=self.rx_params)
            self.ry(qdev, wires=wire, params=self.ry_params)


class QuantumRegression__gen305(tq.QuantumModule):
    """Quantum regression with parameter‑sharing rotation block."""

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.param_layer = ParameterSharedLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.param_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = ["QuantumRegression__gen305", "RegressionDataset", "generate_superposition_data"]
