"""Quantum layer implementation using Pennylane.

Defines `QuantumLayer`, a differentiable variational circuit that
accepts a batch of rotation angles and returns the expectation
values of Pauli‑Z on each qubit.  The circuit consists of
`q_layers` alternating rotation and entangling layers.
"""

import pennylane as qml
import torch
import torch.nn as nn


class QuantumLayer(nn.Module):
    """
    Differentiable quantum layer based on a variational circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    q_layers : int
        Number of alternating rotation and entangling layers.
    """

    def __init__(self, num_qubits: int, q_layers: int = 1) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.q_layers = q_layers
        # Trainable rotation angles: shape (q_layers, num_qubits)
        self.angles = nn.Parameter(torch.randn((q_layers, num_qubits)))
        # Quantum device
        self.dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor) -> torch.Tensor:
            """
            Variational circuit.

            Parameters
            ----------
            inputs : torch.Tensor
                Shape (batch, q_layers, num_qubits) containing rotation angles.
            """
            # Apply alternating layers
            for layer in range(self.q_layers):
                for i in range(self.num_qubits):
                    qml.RY(inputs[layer, i], wires=i)
                # Entangling CNOT chain
                for i in range(self.num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            # Return expectation values of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, q_layers, num_qubits) of rotation angles.

        Returns
        -------
        torch.Tensor
            Shape (batch, num_qubits) of expectation values.
        """
        return self.circuit(inputs)


__all__ = ["QuantumLayer"]
