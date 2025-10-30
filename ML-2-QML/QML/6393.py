"""Hybrid quantum‑classical classifier with quantum kernel.

This module implements the quantum side of the hybrid model. It provides
- a variational circuit that mirrors the classical feed‑forward structure,
- a quantum kernel implemented with TorchQuantum,
- a build_classifier_circuit helper that returns the circuit and metadata
  compatible with the classical helper interface.

The class is fully compatible with the original anchor and can be used
directly as a drop‑in replacement.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

__all__ = ["HybridClassifierKernel", "build_classifier_circuit"]


class HybridClassifierKernel(tq.QuantumModule):
    """Variational quantum circuit with quantum kernel support."""

    def __init__(self, num_qubits: int, depth: int = 2) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.encoding = tq.ParameterVector("x", num_qubits)
        self.weights = tq.ParameterVector("theta", num_qubits * depth)
        self.circuit = self._create_circuit()

    def _create_circuit(self) -> tq.QuantumCircuit:
        qc = tq.QuantumCircuit(self.num_qubits)
        # data encoding
        for qubit in range(self.num_qubits):
            qc.rx(self.encoding[qubit], qubit)
        # variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(self.weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)
        return qc

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor | None = None) -> None:
        """Apply the circuit and optionally compute the quantum kernel."""
        q_device.reset_states(x.shape[0])
        # encode data
        for qubit in range(self.num_qubits):
            func_name_dict["rx"](q_device, wires=[qubit], params=x[:, qubit])
        # variational part
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                func_name_dict["ry"](q_device, wires=[qubit], params=self.weights[idx])
                idx += 1
            for qubit in range(self.num_qubits - 1):
                func_name_dict["cz"](q_device, wires=[qubit, qubit + 1])
        # optional kernel evaluation
        if y is not None:
            idx = 0
            for _ in range(self.depth):
                for qubit in range(self.num_qubits):
                    func_name_dict["ry"](q_device, wires=[qubit], params=-self.weights[idx])
                    idx += 1
                for qubit in range(self.num_qubits - 1):
                    func_name_dict["cz"](q_device, wires=[qubit, qubit + 1])

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the quantum kernel value."""
        device = tq.QuantumDevice(self.num_qubits)
        self.forward(device, x, y)
        return torch.abs(device.states.view(-1)[0])


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[tq.QuantumCircuit, Iterable, Iterable, list]:
    """Return the quantum circuit and metadata for compatibility."""
    model = HybridClassifierKernel(num_qubits, depth)
    encoding = list(range(num_qubits))
    weight_sizes = [model.weights.numel()]
    observables = []  # placeholder
    return model.circuit, encoding, weight_sizes, observables
