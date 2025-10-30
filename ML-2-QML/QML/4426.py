"""Hybrid quantum model that implements a quantum patch filter, a variational classifier, and optional scaling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Iterable, Tuple, List


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[tq.QuantumCircuit, Iterable, Iterable, List[tq.PauliOp]]:
    """Return a simple variational circuit and metadata."""
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
    observables = [tq.PauliZ for _ in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables


class HybridQuanvolutionClassifier(tq.QuantumModule):
    """Quantum counterpart of HybridQuanvolutionClassifier, using a random patch encoder and a variational classifier."""

    def __init__(self, num_qubits: int = 4, depth: int = 2, num_classes: int = 10) -> None:
        super().__init__()
        self.n_wires = num_qubits
        # Encoder mapping image patches to qubits
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Variational classifier circuit
        self.classifier_circuit, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_qubits, depth
        )
        # Classical head to map measurement outcomes to logits
        self.linear = nn.Linear(num_qubits, num_classes)
        self.norm = nn.BatchNorm1d(num_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        measurements: List[torch.Tensor] = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [x[:, r, c], x[:, r, c + 1], x[:, r + 1, c], x[:, r + 1, c + 1]],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.random_layer(qdev)
                m = self.measure(qdev)
                measurements.append(m)
        features = torch.cat(measurements, dim=1)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)
