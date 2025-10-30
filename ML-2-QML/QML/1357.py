import pennylane as qml
import numpy as np

class HybridFCL:
    """
    Quantum feature map used in the hybrid fully connected layer.
    Implements a single‑qubit variational circuit with a parameterized Ry gate
    and returns the expectation value of PauliZ.
    """
    def __init__(self, n_qubits: int = 1, device=None) -> None:
        self.n_qubits = n_qubits
        self.dev = device or qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            qml.Hadamard(wires=range(n_qubits))
            qml.RY(params, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(wires=range(n_qubits)))

        self.circuit = circuit

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Evaluate the quantum circuit for each parameter in `thetas`.

        Args:
            thetas: 1D array of parameters, one per batch element.

        Returns:
            2D array of shape (batch, 1) containing expectation values.
        """
        expectations = np.array([self.circuit(theta) for theta in thetas])
        return expectations.reshape(-1, 1)

__all__ = ["HybridFCL"]
