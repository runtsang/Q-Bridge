"""Hybrid quantum self‑attention regression model.

This module extends the quantum regression seed by embedding a
parameterized self‑attention style circuit.  The circuit consists of
trainable rotations followed by controlled‑X entanglement, mirroring the
classical attention mechanism.  The output is a measurement‑based feature
vector fed into a small classical head.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

# --------------------------------------------------------------------------- #
# Data generation – identical to the classical seed
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic quantum regression data."""
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

# --------------------------------------------------------------------------- #
# Dataset wrapper
# --------------------------------------------------------------------------- #
class RegressionDataset(torch.utils.data.Dataset):
    """Dataset yielding quantum states and scalar targets."""
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
# Quantum self‑attention block
# --------------------------------------------------------------------------- #
class QuantumSelfAttention(tq.QuantumModule):
    """Variational circuit that emulates a self‑attention block."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        # Parameterized single‑qubit rotations
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        # Controlled‑X for entanglement
        self.crx = tq.CRX(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice):
        # Apply rotations to each qubit
        for wire in range(self.num_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)
            self.rz(qdev, wires=wire)
        # Entangle neighboring qubits
        for wire in range(self.num_wires - 1):
            self.crx(qdev, wires=[wire, wire + 1])

# --------------------------------------------------------------------------- #
# Hybrid quantum regression model
# --------------------------------------------------------------------------- #
class HybridQuantumSelfAttentionRegression(tq.QuantumModule):
    """Quantum regression model with attention‑style variational circuit."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.attention = QuantumSelfAttention(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        # Encode the classical data into qubits
        self.encoder(qdev, state_batch)
        # Apply the attention‑style variational circuit
        self.attention(qdev)
        # Extract expectation values as features
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["HybridQuantumSelfAttentionRegression", "RegressionDataset", "generate_superposition_data"]
