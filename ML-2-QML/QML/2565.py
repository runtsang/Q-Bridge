"""Quantum hybrid classifier using torchquantum and a Qiskit circuit builder.

The class `QuantumHybridClassifier` implements a quantum module that mirrors the
classical surrogate.  The `build_classifier_circuit` function provides a Qiskit
circuit with the same parameter layout, enabling direct comparison and hybrid
training.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QuantumHybridClassifier(tq.QuantumModule):
    """Quantum module that encodes 4‑dimensional features and applies a variational ansatz."""

    class QLayer(tq.QuantumModule):
        """A reusable sub‑module inspired by the Quantum‑NAT QLayer."""

        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, num_features: int = 4, depth: int = 3):
        super().__init__()
        self.n_wires = num_features
        # Encoder that maps classical features to a quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

        # Classical post‑processing of the measurement results
        layers = []
        for _ in range(depth):
            layers.append(tq.Linear(self.n_wires, self.n_wires))
        layers.append(tq.Linear(self.n_wires, 2))
        self.classifier = tq.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder, variational layer, measurement and classifier."""
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Assume x has shape [bsz, n_wires] after the classical CNN backbone.
        self.encoder(qdev, x)
        self.q_layer(qdev)
        out = self.measure(qdev)
        out = self.classifier(out)
        return self.norm(out)


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """Construct a Qiskit circuit that matches the torchquantum architecture.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (should match `num_features`).
    depth : int
        Number of variational layers.
    """
    encoding = ParameterVector("x", num_qubits)
    # 8 parameters per layer: 3 random rotations + 3 QLayer ops + 2 entangling gates
    weight_params = ParameterVector("theta", num_qubits * depth * 8)
    circuit = QuantumCircuit(num_qubits)

    # Data re‑uploading encoding (Ry)
    for i, qubit in enumerate(range(num_qubits)):
        circuit.ry(encoding[i], qubit)

    idx = 0
    for _ in range(depth):
        # Random layer: 3 rotations per qubit
        for qubit in range(num_qubits):
            circuit.rz(weight_params[idx], qubit); idx += 1
            circuit.rx(weight_params[idx], qubit); idx += 1
            circuit.ry(weight_params[idx], qubit); idx += 1
        # QLayer‑like operations
        for qubit in range(num_qubits):
            circuit.rx(weight_params[idx], qubit); idx += 1
            circuit.ry(weight_params[idx], qubit); idx += 1
            circuit.rz(weight_params[idx], qubit); idx += 1
        # Entanglement
        for qubit in range(num_qubits - 1):
            circuit.cx(qubit, qubit + 1)
        # Additional gates
        circuit.h(num_qubits - 1)
        circuit.sx(num_qubits - 2)
        circuit.cx(num_qubits - 1, 0)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weight_params), observables


__all__ = ["QuantumHybridClassifier", "build_classifier_circuit"]
