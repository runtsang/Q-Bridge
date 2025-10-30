"""Quantum hybrid regression model that combines a variational circuit with an EstimatorQNN‑style head.
It uses torchquantum for the random layer and parameterized RX/RY gates, and a small H‑Ry‑Rx circuit per qubit that mirrors the EstimatorQNN construction."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate superposition states and labels (complex states)."""
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
    return states.astype(np.complex64), labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset wrapping the superposition states as complex tensors."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class HybridRegressionModel(tq.QuantumModule):
    """Quantum hybrid regressor with a variational circuit and EstimatorQNN head."""
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.h_gate = tq.H()
            self.est_ry = tq.RY(has_params=True, trainable=True)
            self.est_rx = tq.RX(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice, est_ry_params: torch.Tensor, est_rx_params: torch.Tensor):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)
            for wire in range(self.n_wires):
                self.h_gate(qdev, wires=wire)
                self.est_ry(qdev, params=est_ry_params[wire], wires=wire)
                self.est_rx(qdev, params=est_rx_params[wire], wires=wire)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        # Parameters for EstimatorQNN head
        self.est_ry_params = nn.Parameter(torch.randn(num_wires))
        self.est_rx_params = nn.Parameter(torch.randn(num_wires))
        self.measure = tq.MeasureAll(tq.PauliY)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev, self.est_ry_params, self.est_rx_params)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
