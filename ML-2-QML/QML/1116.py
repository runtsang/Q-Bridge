import pennylane as qml
import numpy as np

def EstimatorQNN():
    """Return a quantum neural network that maps 2‑dimensional classical inputs to expectation values."""
    class EstimatorNN:
        def __init__(self, num_qubits: int = 4, entanglement: str = "linear") -> None:
            self.num_qubits = num_qubits
            self.entanglement = entanglement
            self.dev = qml.device("default.qubit", wires=num_qubits)
            weight_shapes = {"weights": num_qubits * 3}
            self.qnode = qml.QNode(
                self._circuit,
                self.dev,
                interface="autograd",
                diff_method="backprop",
                weight_shapes=weight_shapes,
            )

        def _circuit(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
            # Angle encoding of the two classical inputs on the first two qubits
            qml.PhaseShift(inputs[0], wires=0)
            qml.PhaseShift(inputs[1], wires=1)
            # Parameterized rotations on each qubit
            for i in range(self.num_qubits):
                qml.RX(weights[i], wires=i)
                qml.RY(weights[self.num_qubits + i], wires=i)
                qml.RZ(weights[2 * self.num_qubits + i], wires=i)
            # Entanglement pattern
            if self.entanglement == "linear":
                for i in range(self.num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            elif self.entanglement == "full":
                for i in range(self.num_qubits):
                    for j in range(i + 1, self.num_qubits):
                        qml.CNOT(wires=[i, j])
            # Expectation values of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        def __call__(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
            return self.qnode(inputs, weights)

    return EstimatorNN()

__all__ = ["EstimatorNN", "EstimatorQNN"]
