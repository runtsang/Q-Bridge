import pennylane as qml
import numpy as np

class HybridFCL:
    """
    Quantum fully‑connected layer implemented as a variational circuit.
    Parameters are interpreted as rotation angles for a layered Ry
    circuit with CNOT entanglement. The expectation value of the
    Pauli‑Z operator on the last qubit is returned as the output.
    """

    def __init__(self, n_qubits: int = 1, n_layers: int = 1):
        """
        Parameters
        ----------
        n_qubits : int
            Number of qubits in the circuit.
        n_layers : int
            Number of Ry‑CNOT layers.
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(params[layer, i], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(self.n_qubits - 1))

        self.circuit = circuit

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Reshape the flat theta vector into the required shape and evaluate
        the circuit. Returns a single‑element array containing the
        expectation value.
        """
        shape = (self.n_layers, self.n_qubits)
        params = thetas.reshape(shape)
        expectation = self.circuit(params)
        return np.array([expectation])
