"""Quantum self‑attention module using Pennylane."""

import pennylane as qml
import numpy as np


class SelfAttentionModule:
    """
    Variational self‑attention circuit.
    Encodes each token into a qubit via Ry rotations, applies entangling CXs,
    and measures to generate attention scores.
    """

    def __init__(self, num_qubits: int, num_layers: int = 2, device_name: str = "default.qubit"):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device(device_name, wires=num_qubits)

        # Trainable parameters for rotation gates
        self.rot_params = np.random.uniform(0, 2 * np.pi, size=(num_layers, num_qubits, 3))
        # Entangling parameters for controlled rotations
        self.ent_params = np.random.uniform(0, 2 * np.pi, size=(num_layers, num_qubits - 1))

        self.qnode = qml.QNode(self._circuit, self.dev, interface="autograd")

    def _circuit(self, inputs: np.ndarray, rot_params: np.ndarray, ent_params: np.ndarray):
        # Encode inputs
        for i in range(self.num_qubits):
            qml.RY(inputs[i], wires=i)

        # Variational layers
        for layer in range(self.num_layers):
            for i in range(self.num_qubits):
                qml.RX(rot_params[layer, i, 0], wires=i)
                qml.RY(rot_params[layer, i, 1], wires=i)
                qml.RZ(rot_params[layer, i, 2], wires=i)

            for i in range(self.num_qubits - 1):
                qml.CRX(ent_params[layer, i], wires=[i, i + 1])

        # Measure in computational basis
        return qml.expval(qml.PauliZ(wires=range(self.num_qubits)))

    def run(self, inputs: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Execute the circuit on the given inputs and return a vector of
        expectation values that can be interpreted as attention scores.
        Args:
            inputs: array of shape (num_qubits,) with token embeddings
            shots: number of shots for sampling (ignored for autograd mode)
        Returns:
            numpy array of shape (num_qubits,) with attention‑like scores
        """
        # Ensure inputs are 1‑D
        inputs = np.array(inputs).reshape(-1)
        scores = self.qnode(inputs, self.rot_params, self.ent_params)
        # Convert to positive probabilities via softmax
        probs = np.exp(scores) / np.exp(scores).sum()
        return probs
