import pennylane as qml
import numpy as np

class QuantumFullyConnectedLayer:
    """
    Variational quantum circuit with depth‑2 entanglement and Pauli‑Z measurement.
    Designed to emulate a fully connected layer in a quantum‑classical hybrid setting.
    """
    def __init__(self, n_qubits: int = 2, dev: qml.Device = None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.dev = dev or qml.device("default.qubit", wires=n_qubits, shots=shots)
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            # Layer 1: parameterized rotations
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)
            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Layer 2: second set of rotations
            for i in range(self.n_qubits):
                qml.RY(params[i + self.n_qubits], wires=i)
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Accepts a list of parameters (length 2*n_qubits) and returns the
        expectation value of Pauli‑Z on qubit 0 as a 1‑element array.
        """
        params = np.array(thetas, dtype=np.float32)
        if params.size!= 2 * self.n_qubits:
            raise ValueError(f"Expected {2 * self.n_qubits} parameters, got {params.size}")
        expectation = self.circuit(params)
        return np.array([expectation])

def FCL() -> QuantumFullyConnectedLayer:
    """
    Factory returning an instance of the quantum fully connected layer.
    """
    return QuantumFullyConnectedLayer()

__all__ = ["FCL", "QuantumFullyConnectedLayer"]
