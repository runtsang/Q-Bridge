"""Hybrid quantum regression model that fuses a quantum self‑attention block
with a quantum fully‑connected layer.

The implementation mirrors the classical architecture above but replaces
the self‑attention and fully‑connected layers with parameterised
quantum circuits.  The encoder uses a GeneralEncoder to map the
classical input state onto a quantum state.  The quantum self‑attention
is built from RX/RY/RZ gates followed by controlled‑X operations.
The fully‑connected layer is a small circuit that applies a random
layer, single‑qubit rotations and a few entangling gates.  Finally,
the circuit is measured and projected onto a linear head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

# ----------------------------------------------------------------------
# Data generation (identical to the classical version)
# ----------------------------------------------------------------------
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic superposition data.

    The function is identical to the original seed but is re‑implemented
    here to keep the module self‑contained.
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
    """Dataset that returns a dictionary with ``states`` and ``target``."""

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
# Quantum self‑attention block
# ----------------------------------------------------------------------
class QuantumSelfAttention(tq.QuantumModule):
    """Quantum self‑attention circuit.

    The circuit applies a trainable RX/RY/RZ rotation on each qubit
    followed by a chain of controlled‑X gates that entangle neighbours.
    """

    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        for i in range(self.n_qubits):
            self.rx(qdev, wires=i)
            self.ry(qdev, wires=i)
            self.rz(qdev, wires=i)
        for i in range(self.n_qubits - 1):
            self.crx(qdev, wires=[i, i + 1])

# ----------------------------------------------------------------------
# Quantum fully‑connected layer
# ----------------------------------------------------------------------
class QuantumFullyConnectedLayer(tq.QuantumModule):
    """Small parameterised circuit that emulates a single‑qubit
    fully‑connected layer from the QML seed.
    """

    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)

# ----------------------------------------------------------------------
# Hybrid quantum regression model
# ----------------------------------------------------------------------
class QuantumRegressionHybrid(tq.QuantumModule):
    """Quantum hybrid model that mirrors the classical architecture.

    1. Encoder maps the classical state into a quantum state.
    2. QuantumSelfAttention captures pairwise correlations.
    3. QuantumFullyConnectedLayer acts as a parameterised circuit.
    4. Measurement yields a feature vector that is linearly projected.
    """

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.attention = QuantumSelfAttention(num_wires)
        self.qfc = QuantumFullyConnectedLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    @tq.static_support
    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.attention(qdev)
        self.qfc(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["QuantumRegressionHybrid", "RegressionDataset", "generate_superposition_data"]
