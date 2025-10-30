"""Hybrid quantum classifier that mirrors the classical pipeline.

The quantum module uses a CNN feature extractor, a GeneralEncoder to map
classical features onto qubits, a compact QLayer (random & parametric
gates), and a simple variational ansatz.  The `build_classifier_circuit`
helper returns a parameterised variational circuit that can be used with
Qiskit simulators or hardware backends, maintaining the original
interface while incorporating the new quantum layer.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


class HybridClassifier(tq.QuantumModule):
    """Quantum hybrid classifier inspired by QuantumClassifierModel and QuantumNAT."""

    class QLayer(tq.QuantumModule):
        """Compact quantum layer with random and parametric gates."""

        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(self.n_wires)))
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

    def __init__(self, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = 4
        # Feature extractor identical to the classical part of QuantumNAT
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Pool and flatten to obtain 16 features
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        # Encode classical features into qubits
        self.encoder(qdev, pooled)
        # Apply the compact quantum layer
        self.q_layer(qdev)
        # Simple variational ansatz (depth‑controlled)
        for _ in range(self.depth):
            for qubit in range(self.n_wires):
                # Random rotation for demonstration; in practice use trainable params
                tqf.ry(qdev, angles=torch.randn(1, device=x.device), wires=qubit,
                       static=self.static_mode, parent_graph=self.graph)
            for qubit in range(self.n_wires - 1):
                tqf.cz(qdev, wires=[qubit, qubit + 1],
                       static=self.static_mode, parent_graph=self.graph)
        out = self.measure(qdev)
        return self.norm(out)

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
        """Return a parameterised variational circuit with encoding and measurement."""
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

        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
        return circuit, list(encoding), list(weights), observables


__all__ = ["HybridClassifier"]
