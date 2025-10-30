"""Quantum hybrid classifier with optional quantum kernel.

This module builds a reusable class that mirrors the classical interface
while leveraging a variational circuit and a quantum kernel built with
TorchQuantum.  The design follows the original ``QuantumClassifierModel``
seed but adds:
* a depth‑controlled ansatz with data encoding and entangling gates,
* a classical linear read‑out on top of measurement results,
* an optional quantum‑based RBF‑style kernel for kernel‑based methods.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict
import numpy as np


class _QuantumKernalAnsatz(tq.QuantumModule):
    """Quantum kernel ansatz that encodes two samples and returns an overlap."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        # Create a simple data‑encoding circuit
        self.encoding = [
            {"input_idx": [i], "func": "ry", "wires": [i]}
            for i in range(n_wires)
        ]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        # Encode x
        for info in self.encoding:
            params = x[:, info["input_idx"]]
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        # Encode -y
        for info in reversed(self.encoding):
            params = -y[:, info["input_idx"]]
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class _QuantumKernel(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap of two encoded states."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = _QuantumKernalAnsatz(self.n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        # Return absolute overlap
        return torch.abs(self.q_device.states.view(-1)[0])


class HybridClassifier(nn.Module):
    """Quantum hybrid classifier with optional quantum kernel."""
    def __init__(
        self,
        num_qubits: int,
        depth: int = 2,
        use_kernel: bool = False,
    ) -> None:
        super().__init__()
        self.use_kernel = use_kernel
        # Build variational ansatz
        self.encoding = tq.ParameterVector("x", num_qubits)
        self.weights = tq.ParameterVector("theta", num_qubits * depth)
        self.circuit = tq.QuantumCircuit(num_qubits)
        # Data encoding
        for qubit in range(num_qubits):
            self.circuit.rx(self.encoding[qubit], qubit)
        # Variational layers
        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                self.circuit.ry(self.weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                self.circuit.cz(qubit, qubit + 1)
        # Classical read‑out layer
        self.readout = nn.Linear(num_qubits, 2)
        # Optional quantum kernel
        self.kernel = _QuantumKernel(num_qubits) if use_kernel else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits from measurement results."""
        # Prepare device
        q_device = tq.QuantumDevice(n_wires=self.circuit.n_wires)
        # Execute circuit
        self.circuit(self.q_device, x)
        # Expectation values of Z on each qubit
        z_exp = self.circuit.expectation(self.q_device, "Z")
        # Pass through classical read‑out
        return self.readout(z_exp)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """Compute Gram matrix between two sets of samples using the quantum kernel."""
        if self.kernel is None:
            raise RuntimeError("Quantum kernel is not enabled for this model.")
        kernel_tensor = self.kernel(a, b)
        return kernel_tensor.detach().cpu().numpy()

__all__ = ["HybridClassifier"]
