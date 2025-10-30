"""Quantum regression model with a variational circuit, multi‑basis measurement and a small classical head."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create superposition states |ψ(θ,φ)⟩ = cosθ|0…0⟩ + e^{iφ} sinθ|1…1⟩
    with target y = sin(2θ) cosφ, plus optional label noise.
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
    noise = np.random.normal(0, 0.02, size=labels.shape)
    return states, (labels + noise).astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset returning quantum state vectors and noisy target labels.
    """
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
    """
    Variational quantum circuit with entangling layers and multi‑basis measurement,
    followed by a lightweight classical regression head.
    """
    class QLayer(tq.QuantumModule):
        """
        Entangling block: a repeat of CNOT‑RX‑RY layers with trainable parameters.
        """
        def __init__(self, num_wires: int, depth: int = 3):
            super().__init__()
            self.n_wires = num_wires
            self.depth = depth
            self.cnot_layers = nn.ModuleList([tq.CNOT(wires=[i, (i + 1) % num_wires]) for i in range(num_wires)])
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            for _ in range(self.depth):
                self.cnot_layers(qdev)
                for w in range(self.n_wires):
                    self.rx(qdev, wires=w)
                    self.ry(qdev, wires=w)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires, depth=4)
        # Measure in Z, X, Y bases to extract richer features
        self.measure_z = tq.MeasureAll(tq.PauliZ)
        self.measure_x = tq.MeasureAll(tq.PauliX)
        self.measure_y = tq.MeasureAll(tq.PauliY)
        self.head = nn.Linear(3 * num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        feat_z = self.measure_z(qdev)
        feat_x = self.measure_x(qdev)
        feat_y = self.measure_y(qdev)
        features = torch.cat([feat_z, feat_x, feat_y], dim=-1)
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
