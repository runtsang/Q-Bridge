"""Hybrid quantum classifier built on torchquantum.

The quantum block mirrors the classical architecture but replaces dense layers
with a variational ansatz that encodes the input and applies entangling gates.
The observables are a set of Pauli‑Z operators, one per qubit, matching the
output dimensionality of the classical head.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.quantum as tq
import torch.quantum.functional as tqf


class HybridClassifier(tq.QuantumModule):
    """
    Quantum‑enhanced classifier that can be used interchangeably with the classical
    :class:`HybridClassifier`.  It inherits from :class:`torchquantum.QuantumModule`
    and exposes the same public metadata attributes.

    The circuit consists of:
    • A linear‑encoding layer that maps each input feature to an RX rotation.
    • A stack of depth‑controlled variational layers (RY + CZ entanglement).
    • A measurement of Pauli‑Z on each qubit.
    """

    def __init__(self, num_qubits: int, depth: int = 4) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.encoding = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(num_qubits)
            ]
        )
        self.weights = nn.Parameter(
            torch.randn(num_qubits * depth), requires_grad=True
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(num_qubits)

        # Metadata to match the classical API
        self.encoding_indices = list(range(num_qubits))
        self.weight_sizes = [self.weights.numel()]
        self.observables = list(range(num_qubits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_qubits, bsz=bsz, device=x.device, record_op=True)

        # Encode input features as RX rotations
        self.encoding(qdev, x)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for wire in range(self.num_qubits):
                tq.RY(has_params=True, trainable=True)(qdev, wires=[wire], params=self.weights[idx : idx + 1])
                idx += 1
            for wire in range(self.num_qubits - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])

        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["HybridClassifier"]
