import pennylane as qml
import pennylane.numpy as np

class FCL:
    """
    Quantum fully connected layer implemented with a parameterized circuit.
    Supports multiple qubits, entanglement layers, and outputs expectation
    values of PauliZ for each qubit.
    """
    def __init__(self, n_qubits: int, layers: int = 1,
                 backend: str = "default.qubit", shots: int = 1024):
        self.n_qubits = n_qubits
        self.layers = layers
        self.dev = qml.device(backend, wires=n_qubits, shots=shots)
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            # Input encoding
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)
            # Variational layers
            for l in range(self.layers):
                for i in range(self.n_qubits):
                    qml.RZ(params[self.n_qubits + l * self.n_qubits + i], wires=i)
                # Entanglement pattern
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            # Expectation values of PauliZ per qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        self.circuit = circuit

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the circuit with the given parameter vector.
        The parameter vector length must be n_qubits + layers * n_qubits.
        Returns an array of expectation values for each qubit.
        """
        return np.array(self.circuit(thetas))

__all__ = ["FCL"]
