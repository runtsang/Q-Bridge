import pennylane as qml
import numpy as np

class EstimatorQNN:
    """
    Variational quantum circuit for regression with two qubits.
    The circuit is parameterized by input features and trainable weights.
    An expectation value of PauliZ on qubit 0 is returned as the output.
    """

    def __init__(self,
                 device_name: str = "default.qubit",
                 wires: int = 2,
                 num_layers: int = 3):
        self.dev = qml.device(device_name, wires=wires)
        self.num_layers = num_layers

        # Define trainable weight parameters
        self.w = np.random.randn(num_layers, wires, 3)  # RX, RY, RZ per layer

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> float:
            # Encode inputs into qubit 0 with RX
            qml.RX(inputs[0], wires=0)
            qml.RX(inputs[1], wires=1)

            # Variational layers
            for layer in range(num_layers):
                for q in range(wires):
                    qml.RX(weights[layer, q, 0], wires=q)
                    qml.RY(weights[layer, q, 1], wires=q)
                    qml.RZ(weights[layer, q, 2], wires=q)
                # Entanglement
                for q in range(wires - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[wires - 1, 0])

            # Measurement
            return qml.expval(qml.PauliZ(0))

        self.qnode = circuit

    def __call__(self, inputs: np.ndarray, weights: np.ndarray) -> float:
        return self.qnode(inputs, weights)

    def gradient(self, inputs: np.ndarray, weights: np.ndarray):
        """
        Compute the gradient of the circuit output with respect to the trainable weights.
        Returns a numpy array of the same shape as self.w.
        """
        return qml.grad(self.qnode)(inputs, weights)

__all__ = ["EstimatorQNN"]
