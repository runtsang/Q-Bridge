"""
Class SamplerQNNGen072: Quantum sampler network using PennyLane.

Features:
- Parameterized circuit with 2 input angles and 4 trainable weights.
- Entanglement via CNOT gates.
- Returns probability distribution over computational basis states.
"""

import pennylane as qml
import numpy as np

class SamplerQNNGen072:
    def __init__(self, device=None):
        self.dev = device or qml.device("default.qubit", wires=2)
        # Trainable weights initialized randomly
        self.weights = np.random.uniform(0, 2*np.pi, size=4)

        @qml.qnode(self.dev)
        def circuit(inputs, weights):
            # Input rotations
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)
            # Entanglement
            qml.CNOT(0, 1)
            # Parameterized rotations
            qml.RY(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(0, 1)
            qml.RY(weights[2], wires=0)
            qml.RY(weights[3], wires=1)
            # Return probabilities for basis states |00>, |01>, |10>, |11>
            return qml.probs(wires=[0, 1])

        self.circuit = circuit

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit for a given 2‑dimensional input vector.
        """
        return self.circuit(np.array(inputs, dtype=float), self.weights)

    def set_weights(self, new_weights: np.ndarray) -> None:
        """
        Update the trainable weights.
        """
        if new_weights.shape!= self.weights.shape:
            raise ValueError("Weight vector must have shape (4,)")
        self.weights = new_weights

__all__ = ["SamplerQNNGen072"]
