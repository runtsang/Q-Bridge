import pennylane as qml
import numpy as np

class EstimatorQNNMod:
    """Variational quantum regressor.

    The circuit encodes two classical inputs via Ry rotations,
    applies a stack of parameterised rotation and CNOT layers,
    and measures the expectation of Pauli‑Z on the first qubit.
    The class is designed to be used as a drop‑in replacement
    for the classical EstimatorQNNMod in training loops.
    """
    def __init__(self, wires: int = 2, layers: int = 2, shots: int = 1024) -> None:
        self.wires = wires
        self.layers = layers
        self.device = qml.device("default.qubit", wires=wires, shots=shots)
        self.weights = np.random.randn(layers, wires, 3)

        @qml.qnode(self.device, interface="autograd")
        def circuit(inputs, weights):
            # Data encoding
            for i in range(wires):
                qml.RY(inputs[i], wires=i)
            # Parameterised variational layers
            for l in range(layers):
                for w in range(wires):
                    qml.Rot(weights[l, w, 0], weights[l, w, 1], weights[l, w, 2], wires=w)
                # Entangling layer
                for i in range(wires - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(wires=0))

        self.circuit = circuit

    def __call__(self, inputs: np.ndarray) -> float:
        """Evaluate the circuit on a 1‑D array of two features."""
        return float(self.circuit(inputs, self.weights))

    def set_weights(self, new_weights: np.ndarray) -> None:
        """Replace the variational parameters."""
        self.weights = new_weights

__all__ = ["EstimatorQNNMod"]
