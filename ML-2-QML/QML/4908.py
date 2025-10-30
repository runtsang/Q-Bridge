"""Hybrid regression model with quantum feature extraction, self‑attention and
fully‑connected quantum layer.

The implementation follows the classical counterpart but replaces the
feed‑forward head, self‑attention and fully‑connected module with
parameterised quantum circuits.  All components are TorchQuantum
modules so that the model remains trainable with gradient‑based
optimisation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum import QuantumModule

# Data generation and dataset – identical helper as the classical side
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

class HybridRegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Quantum sub‑modules
# --------------------------------------------------------------------------- #
class QSelfAttention(QuantumModule):
    """Quantum self‑attention block inspired by the Qiskit implementation."""
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        # Encode rotation parameters
        for qubit in range(self.n_qubits):
            self.rx(qdev, wires=qubit)
            self.ry(qdev, wires=qubit)
            self.rz(qdev, wires=qubit)
        # Entangle adjacent qubits
        for qubit in range(self.n_qubits - 1):
            self.crx(qdev, wires=[qubit, qubit + 1])
        return self.measure(qdev)

class QFullyConnectedLayer(QuantumModule):
    """Parameterised quantum circuit that mimics a fully‑connected layer."""
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_qubits)))
        self.ry = tq.RY(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.random_layer(qdev)
        for q in range(self.n_qubits):
            self.ry(qdev, wires=q)
        return self.measure(qdev)

class QFraudDetection(QuantumModule):
    """Quantum analogue of the photonic fraud‑detection style encoder."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.random_layer(qdev)
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)
        return self.measure(qdev)

# --------------------------------------------------------------------------- #
# Hybrid regression model – quantum backbone
# --------------------------------------------------------------------------- #
class HybridRegressionModel(QuantumModule):
    """
    Quantum‑centric hybrid regression model.

    Architecture:
    1. General encoder that maps classical state vectors to a quantum device.
    2. QSelfAttention block producing entangled features.
    3. QFraudDetection block that emulates the photonic fraud‑detection style
       encoder (random layer + trainable rotations).
    4. QFullyConnectedLayer that acts as a learnable quantum fully‑connected
       sub‑module.
    5. Classical linear head that maps the concatenated feature vector to a
       scalar output.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Classical‑to‑quantum encoder
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        # Quantum sub‑modules
        self.attention = QSelfAttention(num_wires)
        self.fraud = QFraudDetection(num_wires)
        self.fcl = QFullyConnectedLayer(2)
        # Classical head
        self.head = nn.Linear(num_wires + 2, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)

        # Encode classical data
        self.encoder(qdev, state_batch)

        # Quantum feature extraction
        attn_feat = self.attention(qdev)
        fraud_feat = self.fraud(qdev)
        fcl_feat = self.fcl(qdev)

        # Concatenate all quantum features
        features = torch.cat([attn_feat, fraud_feat, fcl_feat], dim=-1)
        return self.head(features).squeeze(-1)

__all__ = [
    "HybridRegressionModel",
    "HybridRegressionDataset",
    "generate_superposition_data",
]
