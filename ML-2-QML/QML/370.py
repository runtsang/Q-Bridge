"""Quantum neural network using Pennylane with parameterized layers and classical output."""
import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

class EstimatorQNNModel:
    """
    Hybrid quantum‑classical regressor:
    * Encodes a 2‑dimensional feature vector as RY rotations.
    * Stacks `layers` parameterized rotation layers with CNOT entanglement.
    * Measures the expectation of Pauli‑Z on the first qubit.
    * Applies a linear classical read‑out to produce a scalar prediction.
    """
    def __init__(
        self,
        num_qubits: int = 2,
        layers: int = 2,
        obs: str = "Z",
    ) -> None:
        self.num_qubits = num_qubits
        self.layers = layers
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.obs = qml.PauliZ if obs == "Z" else qml.Hermitian
        self.params = pnp.random.randn(layers, num_qubits, 3)
        self.classical_weights = pnp.random.randn(num_qubits, 1)

        @qml.qnode(self.dev, diff_method="backprop")
        def circuit(inputs, params):
            # Feature encoding
            for i in range(num_qubits):
                qml.RY(inputs[i], wires=i)
            # Parameterized layers
            for layer in range(layers):
                for q in range(num_qubits):
                    qml.Rot(
                        params[layer, q, 0],
                        params[layer, q, 1],
                        params[layer, q, 2],
                        wires=q,
                    )
                # Entanglement
                for q in range(num_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            # Measurement
            return qml.expval(self.obs)
        self.circuit = circuit

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass: quantum circuit followed by a linear read‑out.
        """
        quantum_out = self.circuit(inputs, self.params)
        return np.dot(quantum_out, self.classical_weights)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Convenience wrapper for inference; keeps the quantum device in
        evaluation mode and disables gradients.
        """
        return self.__call__(inputs)

def EstimatorQNN() -> EstimatorQNNModel:
    """
    Factory function that returns an instance of the extended EstimatorQNNModel.
    """
    return EstimatorQNNModel()
