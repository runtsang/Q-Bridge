"""Quantum hybrid model that couples a quanvolution filter with a variational classifier.
The quantum part uses torchquantum for the filter and a simple two‑qubit circuit that
mirrors the classical feed‑forward network.  A qiskit builder is also provided for
experiments that require a native qiskit circuit."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
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
        patches: List[torch.Tensor] = []
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

class QuanvolutionHybridClassifier(tq.QuantumModule):
    """Hybrid quantum‑classical classifier that uses the quanvolution filter
    followed by a two‑qubit variational circuit.  The circuit is designed to
    mirror the classical feed‑forward network defined in the ML counterpart.  A
    qiskit circuit builder is also provided for comparative studies."""
    def __init__(self, num_qubits: int = 2, depth: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.num_qubits = num_qubits
        self.depth = depth
        # Classical linear head that maps the 2‑qubit expectation values to logits
        self.linear = nn.Linear(num_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract quantum kernel features
        features = self.qfilter(x)  # shape: (batch, 4*14*14)
        # Use the first two measurement results as rotation angles
        angles = features[:, :self.num_qubits]  # shape: (batch, num_qubits)
        bsz = angles.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.num_qubits, bsz=bsz, device=device)

        # Encoding: RZ gates with the angles
        for qubit in range(self.num_qubits):
            qdev.rz(angles[:, qubit], qubit)

        # Variational layer: simple depth‑wise circuit
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qdev.ry(torch.rand(1).item(), qubit)  # placeholder random parameter
            for qubit in range(self.num_qubits - 1):
                qdev.cz(qubit, qubit + 1)

        # Measure expectation values of Pauli‑Z on each qubit
        exp_vals = torch.stack([qdev.expectation(tq.PauliZ, [q]) for q in range(self.num_qubits)], dim=1)
        logits = self.linear(exp_vals)
        return F.log_softmax(logits, dim=-1)

    @staticmethod
    def build_qiskit_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """Return a qiskit circuit that mirrors the quantum ansatz used in the
        hybrid model.  The circuit is parameterised with an encoding and a
        variational depth."""
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

__all__ = ["QuanvolutionHybridClassifier"]
