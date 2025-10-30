"""Hybrid binary classifier with a quantum variational head.

The module re‑uses the same classical encoder size as the pure‑classical
module but replaces the final dense head with a variational circuit built
with torchquantum.  It also contains a small data generator that mirrors
the superposition dataset from the regression example, which can be used
to produce synthetic classification data by thresholding the regression
labels.

Key features:
- General Ry‑encoding of classical features.
- Random layer followed by trainable RX/RY gates (variational layer).
- Measurement of Pauli‑Z on all qubits and a linear read‑out.
- Optional estimation of a simple 1‑qubit EstimatorQNN circuit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import numpy as np


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Return states of the form cos(theta)|0…0> + exp(i phi) sin(theta)|1…1>.

    The labels are sin(2 theta) * cos(phi), mirroring the regression example.
    They can be thresholded to create a binary classification task.
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


class EstimatorQNN(tq.QuantumModule):
    """1‑qubit variational circuit that mirrors the EstimatorQNN example.

    The circuit applies an H, a data‑dependent Ry, and a trainable Rx.
    """

    def __init__(self) -> None:
        super().__init__()
        self.h = tq.H()
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rx = tq.RX(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.h(qdev, wires=0)
        self.ry(qdev, wires=0)
        self.rx(qdev, wires=0)


class HybridBinaryClassifier(tq.QuantumModule):
    """Quantum‑variational binary classifier.

    The classical embedding is encoded via a general Ry‑encoder,
    processed by a random layer and trainable RX/RY gates, and read out
    with a linear head.
    """

    def __init__(self, num_qubits: int, num_features: int):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_features = num_features

        # Encode each feature into a Ry rotation on a separate qubit
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_qubits}xRy"])

        # Variational layer
        self.q_layer = self.QLayer(num_qubits)

        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Read‑out head
        self.head = nn.Linear(num_qubits, 1)

    class QLayer(tq.QuantumModule):
        """Random layer + trainable RX/RY per qubit."""

        def __init__(self, num_qubits: int):
            super().__init__()
            self.n_wires = num_qubits
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_qubits)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return probability distribution over {0,1}."""
        bsz = features.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_qubits, bsz=bsz, device=features.device)

        # Encode classical features
        self.encoder(qdev, features)

        # Variational processing
        self.q_layer(qdev)

        # Read‑out and read‑out head
        out = self.measure(qdev)
        out = self.head(out).squeeze(-1)

        # Sigmoid to obtain probability
        prob = torch.sigmoid(out)
        return torch.cat((prob, 1 - prob), dim=-1)


__all__ = ["generate_superposition_data", "EstimatorQNN", "HybridBinaryClassifier"]
