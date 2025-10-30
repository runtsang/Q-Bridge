"""Quantum regression model that combines a classical convolution‑like encoder with a variational circuit.

The module uses torchquantum to encode classical features, apply a random
layer plus trainable rotations, measure expectation values, and finally
output a regression prediction.  It can be swapped for a pure classical
model by setting ``use_qiskit=False``."""
import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate quantum superposition states and corresponding labels."""
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

class RegressionDataset(Dataset):
    """Dataset yielding quantum states and target values."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QCNNModel(tq.QuantumModule):
    """Classical encoder that produces a feature vector for the quantum device."""
    def __init__(self, in_features: int = 8, out_features: int = 4) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(in_features, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class QLayer(tq.QuantumModule):
    """Variational layer with a random circuit and trainable rotations."""
    def __init__(self, num_wires: int) -> None:
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

class HybridQuantumRegression(tq.QuantumModule):
    """Hybrid regression model: classical encoder → variational quantum circuit → linear head."""
    def __init__(self, num_features: int, num_wires: int) -> None:
        super().__init__()
        # Classical encoder that outputs a vector of length ``num_wires``.
        self.encoder_model = QCNNModel(in_features=num_features, out_features=num_wires)
        # Quantum encoder that maps the classical vector onto the device.
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=bsz, device=state_batch.device)
        # Classical feature extraction
        encoded = self.encoder_model(state_batch)
        # Encode onto quantum device
        self.encoder(qdev, encoded)
        # Variational layer
        self.q_layer(qdev)
        # Measurement
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["RegressionDataset", "generate_superposition_data", "HybridQuantumRegression"]
