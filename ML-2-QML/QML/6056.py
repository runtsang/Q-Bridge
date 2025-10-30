"""Hybrid quantum regression model combining a variational circuit with a classical head.

This module builds on the original quantum regression example by:
- Using a custom encoder that maps a classical input vector to rotation angles
  (mirroring EstimatorQNN's input‑parameterised circuit).
- Adding a variational layer consisting of a RandomLayer followed by trainable
  rotations on each qubit.
- Measuring all qubits in the Pauli‑Z basis and feeding the expectation values
  to a linear head.
- The architecture is fully compatible with the classical anchor via the
  alias ``QModel``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset

def generate_quantum_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic quantum states and labels.

    States are a superposition of |0...0⟩ and |1...1⟩ with random phase
    and amplitude.  Labels are a sinusoidal function of the superposition
    angles, providing a non‑linear target for the variational circuit.
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
    return states, labels.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset wrapper for the synthetic quantum regression data."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_quantum_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class HybridRegression(tq.QuantumModule):
    """Variational quantum circuit for regression with a linear read‑out."""
    class QLayer(tq.QuantumModule):
        """Variational layer with trainable rotations and optional RandomLayer."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)

    def __init__(self, num_wires: int, encoder_name: str | None = None):
        super().__init__()
        self.n_wires = num_wires
        # Encoder that maps a classical vector to rotation angles on each qubit
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        ) if encoder_name is None else tq.GeneralEncoder(encoder_name)
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

# Backwards compatibility
QModel = HybridRegression

__all__ = ["HybridRegression", "QModel", "RegressionDataset", "generate_quantum_data"]
