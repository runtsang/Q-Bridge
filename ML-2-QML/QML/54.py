import numpy as np
import pennylane as qml

class FCLExtendedQuantum:
    """
    Quantum variant of the extended fully connected layer.
    Implements a variational circuit with user‑defined depth and qubit count.
    The circuit measures the expectation value of Pauli‑Z on the first qubit.
    """
    def __init__(self,
                 n_qubits: int = 1,
                 n_layers: int = 1,
                 device: qml.Device = None) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = device or qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.device)
        def circuit(params: np.ndarray) -> float:
            # params shape: (n_layers, n_qubits, 2) for RX and RZ angles
            for l in range(self.n_layers):
                for q in range(self.n_qubits):
                    qml.RX(params[l, q, 0], wires=q)
                    qml.RZ(params[l, q, 1], wires=q)
                # Entangling pattern: chain + wrap‑around
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with a flat list of parameters.
        The list length must be n_layers * n_qubits * 2.
        """
        params = np.array(thetas).reshape(self.n_layers, self.n_qubits, 2)
        expectation = self.circuit(params)
        return np.array([expectation])

__all__ = ["FCLExtendedQuantum"]
