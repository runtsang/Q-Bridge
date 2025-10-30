"""Hybrid quantum regression model with variational circuit and self‑attention inspired entanglement.

The module defines:
* generate_superposition_data – synthetic quantum state generator.
* RegressionDataset – PyTorch Dataset that returns complex state vectors.
* QSelfAttentionLayer – quantum block that applies trainable rotations followed
  by a random entangling layer, mirroring a classical self‑attention pattern.
* HybridRegressionModel – QuantumModule that encodes the data, applies the
  self‑attention layer, measures, and maps the expectation values to a scalar.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic quantum states of the form
    cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>.
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
    return states, labels


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that yields complex quantum states and scalar targets."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QSelfAttentionLayer(tq.QuantumModule):
    """Quantum self‑attention block inspired by classical attention.

    Applies a trainable rotation on each wire followed by a random entangling
    layer. The rotation parameters are shared across wires to keep the number
    of trainable parameters modest.
    """

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Trainable rotation gates
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        # Random entanglement layer
        self.entangle = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))

    def forward(self, qdev: tq.QuantumDevice) -> None:
        # Apply the same rotation to every wire
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)
            self.rz(qdev, wires=wire)
        # Entangle the wires
        self.entangle(qdev)


class HybridRegressionModel(tq.QuantumModule):
    """Quantum regression model that integrates a self‑attention inspired
    variational circuit with a classical readout head.
    """

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder that maps the input state to a computational basis
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        # Self‑attention inspired variational block
        self.attention = QSelfAttentionLayer(num_wires)
        # Measurement of Pauli‑Z expectation values
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical linear head mapping expectation values to a scalar
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode the input state
        self.encoder(qdev, state_batch)
        # Apply the attention‑style variational block
        self.attention(qdev)
        # Measure expectation values
        features = self.measure(qdev)
        # Classical readout
        return self.head(features).squeeze(-1)


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
