"""Quantum sampler network using Pennylane with a variational circuit and sampling."""

from __future__ import annotations

import pennylane as qml
import numpy as np

class SamplerQNNGen:
    """
    A quantum sampler network built with Pennylane.
    Parameters
    ----------
    num_qubits : int
        Number of qubits in the variational circuit.
    num_layers : int
        Number of variational layers.
    shots : int
        Number of sampling shots per forward pass.
    """
    def __init__(
        self,
        num_qubits: int = 2,
        num_layers: int = 2,
        shots: int = 1024,
    ) -> None:
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.shots = shots

        # Define the device
        self.dev = qml.device("default.qubit", wires=num_qubits, shots=shots)

        # Parameter placeholders
        self.input_params = qml.numpy.array([0.0] * num_qubits)
        self.weight_params = qml.numpy.array([0.0] * (num_qubits * num_layers))

        # Define the variational circuit
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, weights):
            # Encode inputs as Ry rotations
            for i in range(num_qubits):
                qml.RY(inputs[i], wires=i)
            # Variational layers
            idx = 0
            for _ in range(num_layers):
                for i in range(num_qubits):
                    qml.RY(weights[idx], wires=i)
                    idx += 1
                qml.CNOT(wires=[0, 1])
            return qml.probs(wires=range(num_qubits))

        self.circuit = circuit

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass that returns a probability distribution over measurement outcomes.
        """
        # Ensure inputs shape
        inputs = np.asarray(inputs).reshape(1, -1)
        probs = self.circuit(inputs[0], self.weight_params)
        # Flatten to 1D probability vector
        return probs.reshape(-1)

__all__ = ["SamplerQNNGen"]
