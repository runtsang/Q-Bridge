"""Hybrid quantum model combining classification and regression tasks."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable, Tuple

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """Build a variational circuit with shared encoding and two measurement observables.

    The circuit first applies an `RX` encoding of the input features, then
    iterates `depth` layers of `RY` rotations followed by a chain of `CZ`
    entanglers.  Two sets of observables are returned: a list of single‑qubit
    `Z` Pauli operators for classification, and a global `X` operator for
    regression.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)
    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]
    observables.append(SparsePauliOp("X" * num_qubits))
    return circuit, list(encoding), list(weights), observables

class HybridQuantumClassifierRegressor(tq.QuantumModule):
    """Quantum counterpart to the hybrid classical model.

    The model consists of a data‑encoding circuit, a trainable variational
    layer, and two linear heads that consume the expectation values of
    the measurement operators.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        # heads
        self.cls_head = nn.Linear(num_wires, 2)
        self.reg_head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        cls_logits = self.cls_head(features)
        reg_output = self.reg_head(features)
        return cls_logits, reg_output.squeeze(-1)

__all__ = ["build_classifier_circuit", "HybridQuantumClassifierRegressor"]
