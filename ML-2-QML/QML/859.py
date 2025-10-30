import pennylane as qml
import numpy as np
from typing import Iterable

class FCL:
    """
    Quantum variational circuit emulating a fully connected layer.
    Uses parameterized RY gates, a ring of CNOTs, and measures Pauli‑Z.
    """
    def __init__(self, n_qubits: int = 2, dev: str = "default.qubit", shots: int = 1000):
        self.n_qubits = n_qubits
        self.dev = qml.device(dev, wires=n_qubits, shots=shots)

        @qml.qnode(self.dev)
        def circuit(params):
            for i, wire in enumerate(range(n_qubits)):
                qml.RY(params[i], wires=wire)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            qml.CNOT(wires=[n_qubits-1, 0])  # wrap‑around entanglement
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Run the variational circuit with the supplied parameters.
        Returns a 1‑D NumPy array containing the expectation value.
        """
        if len(thetas)!= self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} parameters, got {len(thetas)}")
        return np.array([self.circuit(np.array(thetas))])

__all__ = ["FCL"]
