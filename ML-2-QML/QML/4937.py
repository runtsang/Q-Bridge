"""Hybrid quantum‑classical classifier – quantum implementation.

The quantum module uses torchquantum to build a feature encoder that
acts like the Quanvolution filter, a QCNN‑style variational ansatz,
and a linear head.  It maintains the same public API as the classical
counterpart and can be used wherever the original quantum helper
was expected.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QFeatureEncoder(tq.QuantumModule):
    """Quanvolution‑style quantum feature encoder for 2×2 image patches."""
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


class QLayer(tq.QuantumModule):
    """Variational QCNN‑style ansatz."""
    def __init__(self, depth: int = 3) -> None:
        super().__init__()
        self.n_wires = 4
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)
        self.depth = depth

    def forward(self, qdev: tq.QuantumDevice) -> None:
        for _ in range(self.depth):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)
                self.rz(qdev, wires=wire)
            self.crx(qdev, wires=[0, 1])


class HybridQuantumClassifier(tq.QuantumModule):
    """Hybrid classifier that combines quantum feature extraction,
    a QCNN‑style variational ansatz, and a classical linear head.
    """
    def __init__(self, num_classes: int = 10, patch_depth: int = 3) -> None:
        super().__init__()
        self.feature_extractor = QFeatureEncoder()
        self.ansatz = QLayer(depth=patch_depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        # Quantum device for the patch encoder
        qdev = tq.QuantumDevice(self.feature_extractor.n_wires, bsz=bsz, device=device)
        # Encode all patches into the same device
        features = self.feature_extractor(x)
        # Apply the variational ansatz
        self.ansatz(qdev)
        # Measurement
        out = self.measure(qdev)
        out_flat = out.view(bsz, -1)
        logits = self.linear(out_flat)
        return F.log_softmax(logits, dim=-1)


def build_classifier_circuit(
    num_qubits: int = 4,
    depth: int = 3,
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Return a qiskit circuit and associated metadata.

    The circuit implements a simple layered ansatz that mirrors the
    variational structure used in the quantum module.
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
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


__all__ = ["HybridQuantumClassifier", "build_classifier_circuit"]
