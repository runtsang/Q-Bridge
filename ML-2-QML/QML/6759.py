"""Hybrid quantum‑classical classifier using Pennylane."""

from __future__ import annotations

from typing import Iterable, Tuple

import pennylane as qml
import torch
import torch.nn as nn


class QuantumClassifierModel(nn.Module):
    """
    A hybrid model that embeds a variational quantum circuit within a
    PyTorch module.  The circuit is parameterised by a learnable weight
    matrix and operates on qubit‑wise encoded features.  Pre‑and
    post‑processing linear layers allow for seamless integration with
    classical optimisation pipelines.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int = 2,
        device: str = "default.qubit",
        wires: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.wires = wires or list(range(num_qubits))

        # Classical preprocessing (feature scaling)
        self.pre = nn.Linear(num_qubits, num_qubits)

        # Learnable weight matrix for the variational layers
        self.weights = nn.Parameter(
            torch.randn(depth, num_qubits) * 0.1
        )

        # Quantum device and circuit
        self.dev = qml.device(device, wires=self.wires)
        self.qnode = qml.QNode(
            self._circuit,
            self.dev,
            interface="torch",
            diff_method="backprop",
        )

        # Classical post‑processing to logits
        self.post = nn.Linear(num_qubits, 2)

    def _circuit(self, x: torch.Tensor, weights: torch.Tensor) -> list[torch.Tensor]:
        # Feature encoding
        for i, qubit in enumerate(self.wires):
            qml.RY(x[i], wires=qubit)

        # Variational layers
        for layer in range(self.depth):
            for qubit in self.wires:
                qml.RZ(weights[layer, qubit], wires=qubit)
            for qubit in range(len(self.wires) - 1):
                qml.CNOT(wires=[qubit, qubit + 1])

        # Return expectation values of Z on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in self.wires]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        q_out = self.qnode(x, self.weights)
        return self.post(q_out)

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int,
        depth: int = 2,
    ) -> Tuple[qml.QuantumCircuit, Iterable, Iterable, list[qml.SparsePauliOp]]:
        """
        Construct a reusable quantum circuit with explicit encoding,
        variational parameters, and measurement observables.  The
        returned tuple mirrors the classical helper for direct comparison.
        """
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        from qiskit.quantum_info import SparsePauliOp

        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        circuit = QuantumCircuit(num_qubits)

        # Encoding
        for param, qubit in zip(encoding, range(num_qubits)):
            circuit.rx(param, qubit)

        # Variational layers
        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        # Observables
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]
        return circuit, list(encoding), list(weights), observables

    def get_params(self) -> torch.Tensor:
        """Return a flattened parameter vector for external optimisers."""
        return self.weights.view(-1)

    def set_params(self, params: torch.Tensor) -> None:
        """Set the variational weights from a flattened vector."""
        self.weights.data = params.view(self.depth, self.num_qubits)

__all__ = ["QuantumClassifierModel"]
