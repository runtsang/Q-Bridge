"""Hybrid quantum regression model with a self‑attention style circuit."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic quantum states |ψ⟩ = cosθ|0…0⟩ + e^{iφ} sinθ|1…1⟩."""
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
    """Dataset wrapping synthetic quantum regression data."""
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
    """Quantum circuit implementing a self‑attention style block."""
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.rotation = tq.RX(has_params=True, trainable=True)
        self.entangle = tq.CRX(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice, rotation_params: torch.Tensor, entangle_params: torch.Tensor):
        # Apply rotations
        for wire in range(self.n_qubits):
            self.rotation(qdev, wires=wire, params=rotation_params[wire])
        # Apply entangling gates
        for wire in range(self.n_qubits - 1):
            self.entangle(qdev, wires=[wire, wire + 1], params=entangle_params[wire])

class HybridQModel(tq.QuantumModule):
    """Quantum regression model that uses a self‑attention style circuit."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder maps classical data into quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        # Self‑attention layer
        self.attention = QuantumSelfAttention(num_wires)
        # Additional variational layer
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical head
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor, rotation_params: torch.Tensor | None = None,
                entangle_params: torch.Tensor | None = None) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode classical data
        self.encoder(qdev, state_batch)
        # Default parameters if not provided
        if rotation_params is None:
            rotation_params = torch.randn(self.n_wires, device=state_batch.device)
        if entangle_params is None:
            entangle_params = torch.randn(self.n_wires - 1, device=state_batch.device)
        # Apply self‑attention circuit
        self.attention(qdev, rotation_params, entangle_params)
        # Random variational layer
        self.random_layer(qdev)
        # Measure and feed to classical head
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["HybridQModel", "RegressionDataset", "generate_superposition_data"]
