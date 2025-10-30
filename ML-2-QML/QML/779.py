"""Quantum sampler network using a variational circuit.

The circuit operates on two qubits.  Input angles are encoded via
RY rotations, followed by a fixed entangling layer and a trainable
layer of RY gates.  The output is a probability distribution over
the computational basis, obtained by a measurement of all qubits.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np

class SamplerQNNGen:
    """Variational sampler implemented with Pennylane."""

    def __init__(self, num_qubits: int = 2, entangler: str = "cz", device_name: str = "default.qubit") -> None:
        self.dev = qml.device(device_name, wires=num_qubits)
        self.num_qubits = num_qubits
        self.entangler = entangler

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
            # Input encoding
            for i in range(num_qubits):
                qml.RY(inputs[i], wires=i)

            # Entangling layer
            for i in range(num_qubits - 1):
                if entangler == "cz":
                    qml.CZ(wires=[i, i + 1])
                elif entangler == "cx":
                    qml.CNOT(wires=[i, i + 1])
                else:
                    raise ValueError(f"Unsupported entangler: {entangler}")

            # Trainable rotation layer
            for i in range(num_qubits):
                qml.RY(weights[i], wires=i)

            # Measurement
            return qml.probs(wires=range(num_qubits))

        self.circuit = circuit

    def forward(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Return probability distribution over 4 basis states."""
        return self.circuit(inputs, weights)

    def sample(self, inputs: np.ndarray, weights: np.ndarray, num_samples: int = 1) -> np.ndarray:
        """Draw samples from the output distribution."""
        probs = self.forward(inputs, weights)
        return np.random.choice(len(probs), size=num_samples, p=probs)

    def get_params(self):
        """Return a placeholder list of trainable parameters."""
        # In a real training loop, you would use optimizers to update 'weights'.
        return ["weights"]
