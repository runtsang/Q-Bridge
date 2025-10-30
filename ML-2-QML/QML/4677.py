"""Quantum regression model that embeds a QCNN ansatz followed by a quantum fully‑connected layer.

The model uses torchquantum for circuit construction and training with autograd.
It mirrors the QCNN logic from the seed examples and adds a quantum
fully‑connected layer for richer feature extraction.
"""

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate quantum states of the form cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>.
    Labels are sin(2*theta)*cos(phi) with added noise.
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
    labels = np.sin(2 * thetas) * np.cos(phis) + 0.05 * np.random.randn(samples)
    return states, labels.astype(np.float32)

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

class QFullyConnectedLayer(tq.QuantumModule):
    """Quantum fully‑connected block used in the hybrid quantum model."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> tq.QuantumDevice:
        self.random_layer(qdev)
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)
        return qdev

class QCNNAnsatz(tq.QuantumModule):
    """QCNN‑style ansatz composed of convolution and pooling layers."""
    def __init__(self, num_qbits: int):
        super().__init__()
        self.num_qbits = num_qbits
        # Convolution layers
        self.conv1 = tq.RandomLayer(n_ops=10, wires=list(range(num_qbits)))
        self.conv2 = tq.RandomLayer(n_ops=8, wires=list(range(num_qbits)))
        self.conv3 = tq.RandomLayer(n_ops=6, wires=list(range(num_qbits)))
        # Pooling layers
        self.pool1 = tq.RandomLayer(n_ops=5, wires=list(range(num_qbits)))
        self.pool2 = tq.RandomLayer(n_ops=4, wires=list(range(num_qbits)))
        self.pool3 = tq.RandomLayer(n_ops=3, wires=list(range(num_qbits)))

    def forward(self, qdev: tq.QuantumDevice) -> tq.QuantumDevice:
        self.conv1(qdev)
        self.pool1(qdev)
        self.conv2(qdev)
        self.pool2(qdev)
        self.conv3(qdev)
        self.pool3(qdev)
        return qdev

class HybridRegressionModel(tq.QuantumModule):
    """End‑to‑end quantum regression model combining QCNN ansatz and a quantum fully‑connected layer."""
    def __init__(self, num_qbits: int = 8, num_fc_wires: int = 4):
        super().__init__()
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_qbits}xRy"])
        self.cnn = QCNNAnsatz(num_qbits)
        self.q_fc = QFullyConnectedLayer(num_fc_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_fc_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.cnn.num_qbits, bsz=bsz, device=state_batch.device)
        # Encode classical data
        self.encoder(qdev, state_batch)
        # Apply QCNN ansatz
        self.cnn(qdev)
        # Apply quantum fully‑connected block
        self.q_fc(qdev)
        # Measure Pauli‑Z expectation values on the fully‑connected wires
        all_meas = self.measure(qdev)
        features = all_meas[:, -self.q_fc.n_wires :]
        return self.head(features).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
