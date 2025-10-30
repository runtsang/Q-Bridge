"""Hybrid quantum regression model that embeds a trainable self‑attention
circuit before the variational layers."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

# --------------------------------------------------------------------------- #
#  Dataset & data‑generation utilities
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic quantum states of the form
    cos(θ)|0…0⟩ + e^{iϕ} sin(θ)|1…1⟩ and a smooth target."""
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
    """Quantum dataset compatible with the classical counterpart."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
#  Quantum self‑attention module
# --------------------------------------------------------------------------- #
def SelfAttention():
    """Return a small quantum self‑attention circuit that can be
    differentiated with respect to its parameters."""
    class QuantumSelfAttention(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            # Trainable rotation parameters (3 per qubit)
            self.rotation = nn.Parameter(torch.randn(n_qubits, 3))
            # Trainable entanglement parameters (n-1 per circuit)
            self.entangle = nn.Parameter(torch.randn(n_qubits - 1))

        def forward(self, qdev: tq.QuantumDevice) -> None:
            # Apply per‑qubit rotations
            for i in range(self.n_qubits):
                tq.RX(has_params=True, trainable=True)(qdev, wires=i, params=self.rotation[i, 0])
                tq.RY(has_params=True, trainable=True)(qdev, wires=i, params=self.rotation[i, 1])
                tq.RZ(has_params=True, trainable=True)(qdev, wires=i, params=self.rotation[i, 2])
            # Entangle adjacent qubits
            for i in range(self.n_qubits - 1):
                tq.CRX(has_params=True, trainable=True)(qdev, wires=[i, i + 1], params=self.entangle[i])

    return QuantumSelfAttention

# --------------------------------------------------------------------------- #
#  Hybrid attention‑based quantum model
# --------------------------------------------------------------------------- #
class HybridAttentionQModel(tq.QuantumModule):
    """Quantum regression model that first applies a trainable
    self‑attention block before the variational layers."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires

        # Encoder that maps the input state into the device
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])

        # Attention block
        self.attention = SelfAttention()(num_wires)

        # Variational layer
        self.q_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))

        # Measurement and classical head
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)

        # Encode the input state
        self.encoder(qdev, state_batch)

        # Apply self‑attention circuit
        self.attention(qdev)

        # Variational layer
        self.q_layer(qdev)

        # Classical feature extraction
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["HybridAttentionQModel", "RegressionDataset", "generate_superposition_data"]
