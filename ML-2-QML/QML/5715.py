"""Hybrid quantum regression model with quantum self‑attention encoder."""
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
    return states, labels

class RegressionDataset(torch.utils.data.Dataset):
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
    """Quantum self‑attention encoder."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.rotation_params = nn.Parameter(torch.randn(num_wires, 3))
        self.entangle_params = nn.Parameter(torch.randn(num_wires - 1))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.cnot = tq.CNOT()
        self.rz_ent = tq.RZ(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice):
        for i in range(self.num_wires):
            self.rx(qdev, params=self.rotation_params[i, 0], wires=i)
            self.ry(qdev, params=self.rotation_params[i, 1], wires=i)
            self.rz(qdev, params=self.rotation_params[i, 2], wires=i)
        for i in range(self.num_wires - 1):
            self.cnot(qdev, wires=(i, i + 1))
            self.rz_ent(qdev, params=self.entangle_params[i], wires=i + 1)

class QLayer(tq.QuantumModule):
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)

class QModel(tq.QuantumModule):
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.self_attention = QuantumSelfAttention(num_wires)
        self.q_layer = QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        tq.amplitude_encoding(state_batch, qdev)
        self.self_attention(qdev)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
