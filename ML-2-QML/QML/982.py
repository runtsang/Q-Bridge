"""Quantum regression module that can be used by the hybrid classical–quantum model.

The module implements a variational circuit with 2‑qubit entangling gates
and a linear head that outputs a feature vector of size equal to the
number of qubits.  This module is importable by the classical code and
can be trained jointly.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

# --------------------------------------------------------------------------- #
# Data generation – identical to the original seed but with a slightly
# different random‑angle distribution for reproducibility.
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = np.random.uniform(0, 2 * np.pi, size=samples)
    phis = np.random.uniform(0, 2 * np.pi, size=samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that mirrors the original seed but uses the new data
    generator.  The data are returned as tensors suitable for
    feeding into a quantum model."""
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
# Quantum module – variational circuit with entangling gates
# --------------------------------------------------------------------------- #
class HybridQuantumModule(tq.QuantumModule):
    """Variational quantum circuit that produces a feature vector.

    The circuit consists of:
        * A random layer with 30 random two‑qubit gates.
        * A trainable RX and RY rotation on each qubit.
        * An entangling layer with CNOTs between neighboring qubits.
    The output is a vector of expectation values of Pauli‑Z on each qubit.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.entangle = tq.CNOT(wires=list(range(num_wires - 1)), control_wires=range(num_wires - 1))

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)
            # Entangle neighboring qubits
            for ctrl, tgt in zip(range(self.n_wires - 1), range(1, self.n_wires)):
                self.entangle(qdev, control_wires=ctrl, target_wires=tgt)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder: map the classical state to a quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return features

__all__ = ["HybridQuantumModule", "RegressionDataset", "generate_superposition_data"]
