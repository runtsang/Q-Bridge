"""Core quantum circuit factory with a hybrid classical head using Pennylane."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import pennylane as qml
import numpy as np
import torch


class QuantumClassifierModel:
    """
    A hybrid quantum‑classical classifier that builds a variational circuit
    and optionally augments it with a classical MLP head.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int = 2,
        use_classical_head: bool = False,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.use_classical_head = use_classical_head

        # Device and QNode
        self.dev = qml.device("default.qubit", wires=self.num_qubits)
        self.circuit = self._build_circuit()
        self.qnode = qml.QNode(self.circuit, self.dev, interface="torch")

        # Classical head if requested
        if self.use_classical_head:
            self.classical_head = torch.nn.Sequential(
                torch.nn.Linear(self.num_qubits, self.num_qubits),
                torch.nn.ReLU(),
                torch.nn.Linear(self.num_qubits, 2),
            )

    def _build_circuit(self) -> qml.QNode:
        """Construct a variational circuit with encoding and entangling layers."""
        @qml.qnode(self.dev, interface="torch")
        def circuit(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # Feature encoding
            for i, wire in enumerate(range(self.num_qubits)):
                qml.RX(x[i], wires=wire)

            # Variational ansatz
            idx = 0
            for _ in range(self.depth):
                for wire in range(self.num_qubits):
                    qml.RY(params[idx], wires=wire)
                    idx += 1
                for wire in range(self.num_qubits - 1):
                    qml.CZ(wire, wire + 1)

            # Measurements
            return [qml.expval(qml.PauliZ(w)) for w in range(self.num_qubits)]

        return circuit

    def get_encoding(self) -> List[int]:
        """Return the encoding mapping for the input qubits."""
        return list(range(self.num_qubits))

    def get_weight_counts(self) -> List[int]:
        """Return the weight count for the variational parameters."""
        return [self.num_qubits * self.depth]

    def forward(self, x: np.ndarray) -> torch.Tensor:
        """
        Execute the circuit on input data and optionally feed the result
        into a classical head.

        Parameters
        ----------
        x : np.ndarray
            Input vector of shape (num_qubits,).
        """
        x_torch = torch.tensor(x, dtype=torch.float32)
        # Random initial parameters for demonstration; in practice these would be learned
        params = torch.randn(self.num_qubits * self.depth, requires_grad=True)
        quantum_out = self.qnode(x_torch, params)

        if self.use_classical_head:
            return self.classical_head(quantum_out)
        return quantum_out

    def hybrid_forward(
        self,
        x: np.ndarray,
        classical_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform a hybrid forward pass combining quantum output with
        classical features through a linear layer.

        Parameters
        ----------
        x : np.ndarray
            Input vector for the quantum circuit.
        classical_features : torch.Tensor
            Classical feature tensor to concatenate.
        """
        quantum_out = self.forward(x)
        combined = torch.cat([quantum_out, classical_features], dim=-1)
        # Simple linear classifier on top of concatenated features
        linear = torch.nn.Linear(quantum_out.shape[-1] + classical_features.shape[-1], 2)
        return linear(combined)


__all__ = ["QuantumClassifierModel"]
