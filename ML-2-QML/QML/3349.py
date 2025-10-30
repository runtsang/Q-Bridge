"""Hybrid quantum convolutional regression model.

This module builds on the quantum regression example and replaces the
classical convolution with a quantum convolution circuit.  The
encoder maps 2‑D data into a state using a Ry rotation per wire; the
quantum layer applies a random unitary followed by trainable RX/RY
rotations, mirroring the structure in the reference `QLayer`.  The
measurement produces a feature vector that is fed into a linear head
to output a scalar regression target.

The module is fully compatible with torchquantum and can be trained
with standard PyTorch optimisers.
"""

import torch
import torch.nn as nn
import torchquantum as tq
import numpy as np
from typing import Tuple


def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return states and labels for the quantum regression task."""
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
    """Dataset for the hybrid quantum regression task."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class HybridConvRegressionQuantum(tq.QuantumModule):
    """Quantum convolution analogue of the classical Conv filter."""

    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
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

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder that applies Ry rotations proportional to the input amplitude.
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_batch: Tensor of shape (batch, 2**num_wires) with complex dtype.

        Returns:
            Tensor of shape (batch,) with regression predictions.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode classical data into quantum state
        self.encoder(qdev, state_batch)
        # Apply quantum convolution layer
        self.q_layer(qdev)
        # Measure to obtain feature vector
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = ["HybridConvRegressionQuantum", "RegressionDataset", "generate_superposition_data"]
