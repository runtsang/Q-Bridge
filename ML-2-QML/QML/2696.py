"""Quantum hybrid model that can be used for classification or regression.

The module exposes a build_classifier_circuit and build_regression_circuit
factory that return a quantum circuit and metadata compatible with the
classical factory.  The core model is a torchquantum module that uses a
parameter‑efficient ansatz and a linear head for the chosen task.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Create a qiskit circuit that mirrors the classical encoding.

    The circuit applies an RX encoding followed by a depth‑controlled
    variational ansatz composed of Ry rotations and CZ entangling gates.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        qc.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables

def build_regression_circuit(num_wires: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Create a qiskit circuit for regression with a simple variational ansatz.

    The circuit encodes the input via parameterised RX gates and then
    applies a depth‑controlled layer of Ry rotations and CZ entangling.
    """
    encoding = ParameterVector("x", num_wires)
    weights = ParameterVector("theta", num_wires * depth)

    qc = QuantumCircuit(num_wires)
    for param, wire in zip(encoding, range(num_wires)):
        qc.rx(param, wire)

    idx = 0
    for _ in range(depth):
        for wire in range(num_wires):
            qc.ry(weights[idx], wire)
            idx += 1
        for wire in range(num_wires - 1):
            qc.cz(wire, wire + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_wires - i - 1)) for i in range(num_wires)]
    return qc, list(encoding), list(weights), observables

def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Same as in the classical regression seed."""
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

class HybridModel(tq.QuantumModule):
    """Hybrid quantum‑classical model that supports both classification and regression.

    The model contains a quantum encoder, a variational layer, a measurement,
    and a classical linear head.  The task is selected at construction time.
    """
    def __init__(self, num_wires: int, depth: int, task: str = "classification"):
        super().__init__()
        self.n_wires = num_wires
        self.task = task

        # Encoder that maps classical features to quantum amplitudes
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self._build_variational_layer(depth)
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical head: 2 outputs for classification, 1 for regression
        self.head = nn.Linear(num_wires, 2 if task == "classification" else 1)

    def _build_variational_layer(self, depth: int) -> tq.QuantumModule:
        class VLayer(tq.QuantumModule):
            def __init__(self, n_wires: int, depth: int):
                super().__init__()
                self.n_wires = n_wires
                self.depth = depth
                self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
                self.rx = tq.RX(has_params=True, trainable=True)
                self.ry = tq.RY(has_params=True, trainable=True)

            def forward(self, qdev: tq.QuantumDevice):
                self.random_layer(qdev)
                for wire in range(self.n_wires):
                    self.rx(qdev, wires=wire)
                    self.ry(qdev, wires=wire)

        return VLayer(num_wires, depth)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = [
    "build_classifier_circuit",
    "build_regression_circuit",
    "generate_superposition_data",
    "HybridModel",
]
