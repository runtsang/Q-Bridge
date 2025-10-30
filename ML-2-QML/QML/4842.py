"""Hybrid regression model – quantum implementation.

This module replaces the classical encoder with a variational quantum
circuit that encodes the input features, applies a random layer of
parametric gates, and measures all qubits.  The measurement
outputs feed a linear head that produces a scalar regression target.
The architecture preserves the public API of the classical
counterpart and is fully compatible with the original
``QuantumRegression.py`` usage.

The quantum block draws heavily from the Quantum‑NAT and
Quantum‑Regression examples: a general encoder, a custom QLayer
with random operations and trainable gates, and a batch‑norm
post‑processing step.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data in quantum state space.

    The states are superpositions of ``|0…0⟩`` and ``|1…1⟩`` with random
    angles ``θ`` and ``φ``.  Labels are ``y = sin(2θ) * cos(φ)``,
    matching the original seed.
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


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset wrapping the quantum regression data."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {"states": torch.tensor(self.states[index], dtype=torch.cfloat),
                "target": torch.tensor(self.labels[index], dtype=torch.float32)}


class HybridRegressionModel(tq.QuantumModule):
    """Quantum‑classical hybrid regression network.

    The encoder maps classical inputs into a qubit state using a
    general Ry‑based encoder.  A dedicated ``QLayer`` applies a
    random circuit of parametric rotations and a few fixed gates,
    then the state is measured in the Pauli‑Z basis.  The resulting
    feature vector is passed through a linear head and batch
    normalisation to produce a scalar prediction.
    """

    class QLayer(tq.QuantumModule):
        """Variational block with random gates and trainable rotations."""

        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            # Random layer of 30 gates
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            # Trainable single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            # Entangling gate
            self.cnot = tq.CNOT()

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)
            # A simple entanglement pattern
            for w in range(self.n_wires - 1):
                self.cnot(qdev, wires=[w, w + 1])

    def __init__(self, num_wires: int = 4):
        super().__init__()
        self.n_wires = num_wires
        # General encoder that maps each input feature to a Ry rotation
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(num_wires)]
        )
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)
        self.norm = nn.BatchNorm1d(1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        device = state_batch.device
        # Quantum device with batch support
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=device, record_op=True)
        # Encode classical data into qubit state
        self.encoder(qdev, state_batch)
        # Variational block
        self.q_layer(qdev)
        # Measurement
        features = self.measure(qdev)
        # Linear head + batchnorm
        out = self.head(features)
        out = self.norm(out).squeeze(-1)
        return out


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
