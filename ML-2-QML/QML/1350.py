import pennylane as qml
import numpy as np

class HybridFCL:
    """
    A variational quantum circuit that emulates a fully‑connected layer.
    The circuit consists of alternating rotation layers and entangling CNOTs.
    Parameters are supplied as a flat list via the ``run`` method.
    """
    def __init__(self, n_qubits: int = 4, layers: int = 2,
                 backend: str = "default.qubit", shots: int = 1000):
        self.n_qubits = n_qubits
        self.layers = layers
        self.dev = qml.device(backend, wires=n_qubits, shots=shots)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            # params shape: (layers, n_qubits, 3) for Rx,Ry,Rz
            for l in range(layers):
                for q in range(n_qubits):
                    qml.RX(params[l, q, 0], wires=q)
                    qml.RY(params[l, q, 1], wires=q)
                    qml.RZ(params[l, q, 2], wires=q)
                # entangle with a simple ladder of CNOTs
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with a supplied flat list of parameters.
        Parameters are reshaped to (layers, n_qubits, 3) before evaluation.
        Returns a 1‑D array containing the expectation value.
        """
        flat = np.asarray(list(thetas), dtype=np.float32)
        expected_size = self.layers * self.n_qubits * 3
        if flat.size!= expected_size:
            raise ValueError(f"Expected {expected_size} parameters, got {flat.size}")
        params = flat.reshape((self.layers, self.n_qubits, 3))
        expval = self.circuit(params)
        return np.array([expval])
