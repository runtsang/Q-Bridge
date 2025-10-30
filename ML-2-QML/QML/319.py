"""Quantum sampler using a Pennylane QNode with parameterised rotations and entanglement."""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from pennylane.measurements import Probability

# Define a device that returns probabilities for all basis states
dev = qml.device("default.qubit", wires=2, shots=1024)

def _build_circuit(input_params: np.ndarray, weight_params: np.ndarray) -> None:
    """
    Internal helper that encodes both input and weight parameters into the circuit.
    """
    # Input encoding with RY rotations
    for i, param in enumerate(input_params):
        qml.RY(param, wires=i)

    # Entangling block
    qml.CNOT(wires=[0, 1])

    # Parameterised rotation layers
    for i, param in enumerate(weight_params):
        qml.RY(param, wires=i)

    # Second entangling block
    qml.CNOT(wires=[0, 1])

def sampler_qnn(input_dim: int = 2, weight_dim: int = 4) -> qml.QNode:
    """
    Returns a QNode that outputs a probability vector over `input_dim` basis states.
    """
    @qml.qnode(dev, interface="autograd")
    def circuit(inputs: np.ndarray, weights: np.ndarray):
        _build_circuit(inputs, weights)
        return qml.probs(wires=range(input_dim))

    return circuit

# Example usage
if __name__ == "__main__":
    # Random initial parameters
    inputs = np.random.uniform(-np.pi, np.pi, size=(2,))
    weights = np.random.uniform(-np.pi, np.pi, size=(4,))
    qnode = sampler_qnn()
    probs = qnode(inputs, weights)
    print("Quantum sampler probabilities:", probs)
