"""Quantum hybrid classifier/regressor using torchquantum.

The implementation follows the same public API as the classical
counterpart.  It provides a Qiskit circuit builder for
classification and regression, and a `QuantumModule` that can be
instantiated with a `regression=True` flag to obtain a single‑output
regression head.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a simple layered ansatz with explicit encoding and variational parameters."""
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

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


def build_regressor_circuit(num_qubits: int, hidden: int = 32) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a shallow ansatz suitable for regression."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * hidden)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(hidden):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


class QuantumClassifierModel(tq.QuantumModule):
    """Quantum module that can perform classification or regression.

    The module wraps a parameter‑efficient ansatz with a classical
    linear head.  Setting ``regression=True`` swaps the two‑class
    softmax into a single‑output regression head, mirroring the
    classical API.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int, depth: int):
            super().__init__()
            self.n_wires = num_wires
            self.depth = depth
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int, depth: int = 3, regression: bool = False):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires, depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.class_head = nn.Linear(num_wires, 2)
        self.regression = regression
        if regression:
            self.reg_head = nn.Linear(num_wires, 1)
        else:
            self.reg_head = None

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        logits = self.class_head(features)
        if self.regression and self.reg_head is not None:
            reg = self.reg_head(features)
            return logits, reg
        return logits

    def get_weight_sizes(self) -> Tuple[List[int], List[int]]:
        """Return weight sizes for classification and regression heads."""
        class_weights = [p.numel() for p in self.class_head.parameters()]
        if self.regression:
            reg_weights = [p.numel() for p in self.reg_head.parameters()]
        else:
            reg_weights = []
        return class_weights, reg_weights


__all__ = ["QuantumClassifierModel", "build_classifier_circuit", "build_regressor_circuit"]
