"""Hybrid quantum regression model that implements QCNN‑style convolutions using torchquantum."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic quantum states of the form
    cos(θ)|0…0⟩ + e^{iφ} sin(θ)|1…1⟩ with a target derived from θ and φ.
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
    """Dataset wrapper that returns quantum states and targets."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class HybridRegressionModel(tq.QuantumModule):
    """
    Quantum regression model that mirrors the classical QCNN layers.
    It uses a feature encoder, a stack of convolutional and pooling blocks
    implemented with `torchquantum` primitives, and a linear read‑out head.
    """

    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_ansatz = self._build_ansatz()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def _build_ansatz(self) -> tq.QuantumModule:
        """Construct a QCNN‑style ansatz with conv and pool layers."""
        layers = [
            self._conv_layer(8, "c1"),
            self._pool_layer(8, "p1"),
            self._conv_layer(4, "c2"),
            self._pool_layer(4, "p2"),
            self._conv_layer(2, "c3"),
            self._pool_layer(2, "p3"),
        ]
        return tq.Sequential(layers)

    def _conv_layer(self, num_qubits: int, name: str) -> tq.QuantumModule:
        """Two‑qubit convolution block: CX → RZ → RY."""
        conv = tq.Sequential()
        for q in range(0, num_qubits, 2):
            conv.append(tq.CX(has_params=False, trainable=False))
            conv.append(tq.RZ(has_params=True, trainable=True))
            conv.append(tq.RY(has_params=True, trainable=True))
        conv.name = name
        return conv

    def _pool_layer(self, num_qubits: int, name: str) -> tq.QuantumModule:
        """Pooling block that reduces circuit depth via parameterized rotations."""
        pool = tq.Sequential()
        for q in range(0, num_qubits, 2):
            pool.append(tq.RZ(has_params=True, trainable=True))
            pool.append(tq.RY(has_params=True, trainable=True))
        pool.name = name
        return pool

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Execute the quantum circuit and read out a regression prediction."""
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_ansatz(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
