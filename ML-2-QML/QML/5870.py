"""Hybrid quantum classifier: quantum quanvolution feature extractor followed by a variational ansatz."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torchquantum as tq
from qiskit.quantum_info import SparsePauliOp


class QuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class HybridQuantumClassifierQML(tq.QuantumModule):
    """Quantum hybrid classifier: quanvolution feature extractor + variational ansatz."""
    def __init__(self, num_qubits: int = 14, depth: int = 3) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        # Variational circuit parameters
        self.num_qubits = num_qubits
        self.depth = depth
        self.encoding = tq.ParameterVector("x", num_qubits)
        self.weights = tq.ParameterVector("theta", num_qubits * depth)
        self.circuit = self._build_ansatz()

    def _build_ansatz(self) -> tq.QuantumCircuit:
        qc = tq.QuantumCircuit(self.num_qubits)
        for param, qubit in zip(self.encoding, range(self.num_qubits)):
            qc.rx(param, qubit)
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(self.weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)  # shape (batch, 4*14*14)
        # Map classical features to qubit states via a simple linear layer
        # (implemented as a parameterized rotation)
        param_map = tq.ParameterVector("phi", self.num_qubits)
        qdev = tq.QuantumDevice(self.num_qubits, bsz=x.shape[0], device=x.device)
        for i in range(self.num_qubits):
            qdev.rx(param_map[i] * features[:, i], i)  # simple encoding
        # Apply variational ansatz
        self.circuit(qdev)
        # Measure in Z basis
        return tq.measure(qdev, tq.PauliZ)

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[tq.QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """
    Construct a quantum circuit with explicit encoding, variational parameters,
    and measurement observables mirroring the classical interface.
    """
    encoding = tq.ParameterVector("x", num_qubits)
    weights = tq.ParameterVector("theta", num_qubits * depth)
    circuit = tq.QuantumCircuit(num_qubits)
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


__all__ = ["HybridQuantumClassifierQML", "build_classifier_circuit"]
