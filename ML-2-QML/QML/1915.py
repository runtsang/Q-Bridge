import pennylane as qml
from pennylane import numpy as np

class EstimatorQNNGen:
    """
    Hybrid variational quantum‑neural network for regression.
    The circuit consists of alternating rotation layers and entangling CNOTs.
    """
    def __init__(
        self,
        num_qubits: int = 4,
        layers: int = 3,
        weights: np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.num_qubits = num_qubits
        self.layers = layers
        self._weights = weights
        if seed is not None:
            np.random.seed(seed)
        if self._weights is None:
            self._weights = 0.01 * np.random.randn(layers, num_qubits, 3)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray):
            # Feature map: encode inputs via RY rotations
            for i in range(num_qubits):
                qml.RY(inputs[i], wires=i)
            # Variational layers
            for l in range(layers):
                for i in range(num_qubits):
                    qml.RX(weights[l, i, 0], wires=i)
                    qml.RY(weights[l, i, 1], wires=i)
                    qml.RZ(weights[l, i, 2], wires=i)
                # Simple entangling pattern
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            # Readout: expectation of PauliZ on first qubit
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit for a batch of inputs.
        """
        return np.array([self.circuit(x, self._weights) for x in inputs])

def EstimatorQNN() -> EstimatorQNNGen:
    """Factory returning a default‑configured QNN instance."""
    return EstimatorQNNGen(num_qubits=4, layers=4, seed=123)

__all__ = ["EstimatorQNNGen", "EstimatorQNN"]
