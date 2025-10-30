"""Quantum classifier that combines a quantum quanvolution filter with a variational circuit."""

from __future__ import annotations

import torch
import torchquantum as tq
import torch.nn as nn
import numpy as np

from qiskit import Aer, execute
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QuanvolutionFilterQuantum(tq.QuantumModule):
    """Quantum 2×2 patch kernel using a random two‑qubit circuit producing 4‑dimensional features."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
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
        # Aggregate across patches to obtain a fixed 4‑dimensional feature vector
        features = torch.stack(patches, dim=1).mean(dim=1)
        return features


class VariationalClassifier(nn.Module):
    """Variational circuit that maps 4‑dimensional features to class logits via expectation values."""
    def __init__(self, num_qubits: int, depth: int, num_classes: int = 10, backend=None) -> None:
        super().__init__()
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth)
        self.backend = backend or Aer.get_backend("statevector_simulator")
        self.num_classes = num_classes
        self.num_qubits = num_qubits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_qubits)
        bound_circuit = self.circuit.bind_parameters({self.encoding[i]: x[:, i] for i in range(self.num_qubits)})
        job = execute(bound_circuit, self.backend)
        result = job.result()
        statevector = result.get_statevector(bound_circuit)
        logits = []
        for obs in self.observables:
            exp_val = np.real(np.vdot(statevector, obs.to_matrix() @ statevector))
            logits.append(exp_val)
        logits = torch.tensor(logits, dtype=torch.float32, device=x.device)
        return logits


class QuantumClassifier(nn.Module):
    """Hybrid network: quantum quanvolution filter + variational classifier."""
    def __init__(self, num_qubits: int = 4, depth: int = 3, num_classes: int = 10, backend=None) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilterQuantum()
        self.vclassifier = VariationalClassifier(num_qubits, depth, num_classes, backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        return self.vclassifier(features)


def build_classifier_circuit(num_qubits: int, depth: int) -> tuple[QuantumCircuit, list[ParameterVector], list[ParameterVector], list[SparsePauliOp]]:
    """
    Construct a variational circuit with data‑uploading ansatz.
    Returns the circuit, encoding parameters, weight parameters, and observables.
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

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables


__all__ = ["QuanvolutionFilterQuantum", "VariationalClassifier", "QuantumClassifier", "build_classifier_circuit"]
