"""Hybrid quantum‑NAT model with a classical post‑processing head.

The quantum core mirrors the classical `HybridQuantumClassifier`:
feature encoding, a variational layer, measurement, and a linear
classifier.  The helper `build_classifier_circuit` returns a Qiskit
circuit and matching metadata for direct comparison with the classical
counterpart.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchquantum as tq
import torchquantum.functional as tqf

class HybridQuantumClassifier(tq.QuantumModule):
    """
    Quantum‑NAT inspired encoder + variational layer + classical classifier.
    """

    class QLayer(tq.QuantumModule):
        """
        A variational block with a random layer, single‑qubit rotations,
        a controlled‑RX, and a small entangling window.
        """

        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, num_features: int = 4, num_classes: int = 2, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = num_features
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)
        self.classifier_head = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Classical pooling to feed into the encoder
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        normed = self.norm(out)
        logits = self.classifier_head(normed)
        return logits


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple["QuantumCircuit", Iterable, Iterable, List["SparsePauliOp"]]:
    """
    Builds a Qiskit circuit that matches the classical feed‑forward
    architecture.  Returns the circuit, encoding parameters, weight
    parameters, and measurement observables.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp

    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    # Data‑encoding layer (RX rotations)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    # Variational blocks
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables


__all__ = ["HybridQuantumClassifier", "build_classifier_circuit"]
