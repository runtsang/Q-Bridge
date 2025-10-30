import pennylane as qml
import numpy as np

class FullyConnectedLayer:
    """
    Variational quantum circuit mimicking a fully‑connected layer.
    Uses a Ry rotation per qubit followed by a linear CNOT chain to
    introduce entanglement.  The circuit returns the expectation of
    Pauli‑Z on wire 0, analogous to the classical output.
    """
    def __init__(self, n_qubits: int = 2, device: str = 'default.qubit', shots: int = 1000):
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=n_qubits, shots=shots)
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev)
        def circuit(params):
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit for a given parameter vector.
        The vector length must match ``n_qubits``.
        """
        if len(thetas)!= self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} theta values, got {len(thetas)}")
        expectation = self.circuit(thetas)
        return np.array([expectation])
