import pennylane as qml
import numpy as np
from typing import List

class QuantumClassifierModel:
    """
    Variational quantum classifier with a data‑encoding layer followed by
    alternating rotation and entanglement blocks.  The circuit is compatible
    with PennyLane's autograd and Qiskit simulators via the ``qml.device`` API.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (equals number of input features).
    depth : int
        Number of variational layers.
    """

    def __init__(self, num_qubits: int, depth: int, device: qml.Device | None = None) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.encoding = np.arange(num_qubits)
        self.weight_sizes = [num_qubits] + [num_qubits] * depth
        self.observables = [qml.PauliZ(i) for i in range(num_qubits)]

        self.device = device or qml.device("default.qubit", wires=num_qubits)

        # Create a flat parameter vector: encoding + variational weights
        total_params = num_qubits + num_qubits * depth
        self.params = np.random.uniform(0, 2 * np.pi, size=total_params)

        @qml.qnode(self.device, interface="autograd")
        def circuit(x: np.ndarray, params: np.ndarray) -> np.ndarray:
            # Data encoding
            for i, wire in enumerate(range(num_qubits)):
                qml.RX(x[i], wire)
            # Variational layers
            idx = num_qubits
            for _ in range(depth):
                for wire in range(num_qubits):
                    qml.RY(params[idx], wire)
                    idx += 1
                for wire in range(num_qubits - 1):
                    qml.CZ(wire, wire + 1)
            # Measurement
            return [qml.expval(o) for o in self.observables]

        self._circuit = circuit

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return the expectation values of the Z observables."""
        return self._circuit(x, self.params)

    def get_parameter_vector(self) -> np.ndarray:
        return self.params.copy()

    def set_parameter_vector(self, vec: np.ndarray) -> None:
        self.params = vec.copy()

    def train_step(
        self,
        x: np.ndarray,
        y: int,
        lr: float = 0.01,
        loss_fn: str = "mse",
    ) -> float:
        """Simple stochastic gradient update using autograd."""
        def loss_fn_wrapper(params: np.ndarray) -> float:
            preds = self._circuit(x, params)
            # Map binary classification to [0,1] via sigmoid of first observable
            prob = 1 / (1 + np.exp(-preds[0]))
            if loss_fn == "mse":
                return (prob - y) ** 2
            else:  # cross‑entropy
                return - (y * np.log(prob) + (1 - y) * np.log(1 - prob))

        grads = qml.grad(loss_fn_wrapper)(self.params)
        self.params -= lr * grads
        return loss_fn_wrapper(self.params)

__all__ = ["QuantumClassifierModel"]
