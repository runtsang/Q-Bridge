import pennylane as qml
import numpy as np

class FullyConnectedLayer:
    """
    Quantum variational circuit that emulates a fully connected layer.
    Uses a simple feature map (RZ) followed by entanglement and Pauli‑Z measurement.
    """
    def __init__(self, n_qubits: int = 1, device=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        if device is None:
            self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
        else:
            self.dev = device

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            # Feature map: encode each param into an RZ gate
            for i in range(self.n_qubits):
                qml.RZ(params[i], wires=i)
            # Entangle
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measure expectation of PauliZ on all qubits
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

        self.circuit = circuit

    def run(self, thetas: np.ndarray | Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with the supplied parameters and return the
        expectation values as a NumPy array.
        """
        params = np.array(list(thetas), dtype=np.float64)
        if params.shape[0]!= self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} parameters, got {params.shape[0]}")
        exps = self.circuit(params)
        return np.array(exps)
