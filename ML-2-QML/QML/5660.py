"""Quantum regression model that mirrors the classical interface but
operates purely in the quantum domain.

The module encodes a real‑valued feature vector into a quantum state,
runs a variational circuit, measures all qubits in the Pauli‑Z basis,
and applies a small classical head to produce a scalar output.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

# --------------------------------------------------------------------------- #
# Data generation – same as in the ML seed
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0…0⟩ + e^{iϕ} sin(theta)|1…1⟩."""
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

# --------------------------------------------------------------------------- #
# Dataset wrapper
# --------------------------------------------------------------------------- #
class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that yields quantum states and target values."""
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
# Quantum regression model
# --------------------------------------------------------------------------- #
class QModel(tq.QuantumModule):
    """Quantum regression model with an encoder, variational layer,
    measurement, and a classical linear head."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder: general Ry rotation per wire
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        # Variational circuit – a RandomLayer with trainable parameters
        self.var_layer = tq.RandomLayer(
            n_ops=30,
            wires=range(num_wires),
            has_params=True,
            trainable=True,
        )
        # Measurement of all qubits in Pauli‑Z
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical head
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Run the quantum circuit and produce a scalar output."""
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.var_layer(qdev)
        features = self.measure(qdev)  # shape (bsz, n_wires)
        return self.head(features).squeeze(-1)

__all__ = [
    "QModel",
    "RegressionDataset",
    "generate_superposition_data",
]
