"""Quantum‑centric quanvolution and classifier using torchquantum and a variational ansatz."""

from __future__ import annotations

import torch
import torchquantum as tq
from typing import Iterable, Tuple

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[tq.QuantumCircuit, Iterable, Iterable, list[tq.Pauli]]:
    """Create a layered variational circuit with explicit encoding and observables."""
    circuit = tq.QuantumCircuit(num_qubits)
    # Encoding layer (fixed for simplicity)
    for qubit in range(num_qubits):
        circuit.rx(0.0, qubit)
    # Variational layers
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(0.5, qubit)
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)
    observables = [tq.PauliZ(i) for i in range(num_qubits)]
    return circuit, [], [], observables

class QuanvolutionHybrid(tq.QuantumModule):
    """Quantum hybrid that applies a quanvolution filter and a variational classifier."""
    def __init__(self, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = 4
        # Quanvolution filter components
        self.encoder = tq.GeneralEncoder([
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ])
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Variational classifier
        self.classifier_circuit, _, _, self.observables = build_classifier_circuit(self.n_wires, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        # 28×28 image → 2×2 patches → quantum measurement
        x = x.view(bsz, 28, 28)
        patches: list[torch.Tensor] = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [x[:, r, c], x[:, r, c + 1], x[:, r + 1, c], x[:, r + 1, c + 1]],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        features = torch.cat(patches, dim=1)

        # Encode the first four feature columns into the variational circuit
        qdev_cls = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        for i in range(self.n_wires):
            qdev_cls.apply(tq.RX(features[:, i], wires=[i]))

        # Apply the learnable variational layers
        qdev_cls.apply(self.classifier_circuit)

        # Final measurement to obtain logits
        logits = tq.MeasureAll(tq.PauliZ)(qdev_cls).view(bsz, self.n_wires)
        return logits

__all__ = ["QuanvolutionHybrid", "build_classifier_circuit"]
