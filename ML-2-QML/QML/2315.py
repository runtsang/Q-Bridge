"""Hybrid quantum model that can perform classification and regression.

The quantum circuit consists of a general encoder, a random layer, and
parameterised RX/RY rotations.  After measurement the outputs are fed
into either a classification head (softmax over two Z‑measurements) or
a regression head (single linear output).  The `build_classifier_circuit`
function reproduces the simple ansatz from the original classification
seed and can be used for rapid prototyping or for generating a
classical reference model.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Same data generator as the classical counterpart but returning complex states."""
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


class SuperpositionDataset(torch.utils.data.Dataset):
    """Quantum version of the dataset used by the classical model."""

    def __init__(self, samples: int, num_wires: int, task: str = "regression"):
        self.states, self.labels = generate_superposition_data(num_wires, samples)
        if task == "classification":
            self.labels = (self.labels > 0).astype(np.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuantumClassifierModel(tq.QuantumModule):
    """Quantum model with a shared encoder and two output heads."""

    class QLayer(tq.QuantumModule):
        """Variational layer combining a random circuit and single‑qubit rotations."""

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

    def __init__(self, num_wires: int, task: str = "classification"):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)

        self.classification_head = nn.Linear(num_wires, 2)
        self.regression_head = nn.Linear(num_wires, 1)

        self.task = task

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        if self.task == "classification":
            return self.classification_head(features)
        return self.regression_head(features).squeeze(-1)

    def switch_task(self, task: str) -> None:
        """Change the active head."""
        assert task in {"classification", "regression"}
        self.task = task

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
        """Return a simple layered ansatz with encoding and variational parameters."""
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


__all__ = ["QuantumClassifierModel", "SuperpositionDataset", "generate_superposition_data"]
