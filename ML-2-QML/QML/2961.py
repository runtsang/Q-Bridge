"""Quantum implementation of the hybrid model, supporting both classification and regression."""
from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torchquantum as tq
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from torch.utils.data import Dataset


def generate_classification_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic binary classification data with a non‑linear decision boundary."""
    X = np.random.randn(samples, num_wires).astype(np.float32)
    y = (np.sum(X ** 2, axis=1) > num_wires).astype(np.int64)
    return X, y


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Superposition data for the regression task."""
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


class ClassificationDataset(Dataset):
    """Dataset wrapper for the synthetic classification data."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_classification_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class RegressionDataset(Dataset):
    """Dataset wrapper for the superposition regression data."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def build_hybrid_circuit(num_qubits: int, depth: int, task: str = "classification") -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a layered variational ansatz with explicit encoding and variational parameters.
    The function returns the circuit together with metadata that matches the classical interface.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables


class HybridQModel(tq.QuantumModule):
    """
    Quantum hybrid model that can perform either classification or regression.
    It implements a variational circuit as a feature extractor followed by a classical head.
    """

    class _VariationalLayer(tq.QuantumModule):
        def __init__(self, num_wires: int, depth: int):
            super().__init__()
            self.n_wires = num_wires
            self.depth = depth
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int, depth: int, task: str = "classification"):
        super().__init__()
        self.n_wires = num_wires
        self.depth = depth
        self.task = task
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self._VariationalLayer(num_wires, depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        if task == "classification":
            self.head = nn.Linear(num_wires, 2)
        else:
            self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode the classical state, run the variational circuit,
        measure all qubits and apply the classical head.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        output = self.head(features)
        return output.squeeze(-1) if self.task == "regression" else output


__all__ = [
    "HybridQModel",
    "ClassificationDataset",
    "RegressionDataset",
    "build_hybrid_circuit",
    "generate_classification_data",
    "generate_superposition_data",
]
