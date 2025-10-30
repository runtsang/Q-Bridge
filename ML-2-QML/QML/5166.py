"""Hybrid quantum model that mirrors the classical HybridNATModel.

This version replaces the CNN backbone with a quantum encoder and
variational circuit.  It uses TorchQuantum for the quantum
operations and retains the same public API so that it can be swapped
into any hybrid training loop.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import hadamard, sx, cnot
from torchquantum.core.circuit import QuantumCircuit
from torchquantum.core.operators import PauliZ


# ----------------------------------------------------------------------
# Data utilities (quantum version)
# ----------------------------------------------------------------------
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate quantum states of the form
    cos(theta)|0…0⟩ + e^{iϕ} sin(theta)|1…1⟩ and corresponding labels.
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
    """Dataset returning a tensor of complex amplitudes and a scalar target."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# ----------------------------------------------------------------------
# Quantum helper: a simple variational layer
# ----------------------------------------------------------------------
class QLayer(tq.QuantumModule):
    """Variational block consisting of a random layer followed by single‑wire rotations."""

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


# ----------------------------------------------------------------------
# Hybrid quantum model
# ----------------------------------------------------------------------
class HybridNATModel(tq.QuantumModule):
    """Quantum counterpart to the classical HybridNATModel.

    It encodes a classical feature vector into a quantum state,
    applies a variational circuit, measures in the computational basis,
    and feeds the resulting expectation values to a classical head.
    """

    def __init__(self, num_wires: int = 4):
        super().__init__()
        self.n_wires = num_wires

        # Encoder: simple Ry rotation for each wire
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])

        # Variational block
        self.q_layer = QLayer(num_wires)

        # Measurement
        self.measure = tq.MeasureAll(PauliZ)

        # Classical head
        self.head = nn.Linear(num_wires, 4)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            state_batch: Tensor of shape (batch, num_wires) containing
                         real‑valued features to be encoded.

        Returns:
            Tensor of shape (batch, 4) after the classical head.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = ["HybridNATModel", "RegressionDataset", "generate_superposition_data"]
