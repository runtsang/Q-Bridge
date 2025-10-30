"""Quantum self‑attention using Pennylane."""
import pennylane as qml
import numpy as np

class SelfAttention:
    """
    Variational self‑attention block. The circuit implements
    a parameterised rotation followed by a controlled‑X entanglement
    pattern, mirroring the classical attention weight computation.
    The output is a probability distribution over the qubits that
    is interpreted as attention scores.
    """
    def __init__(self, n_qubits: int = 4, wires=None):
        self.n_qubits = n_qubits
        self.wires = wires or list(range(n_qubits))
        self.dev = qml.device("default.qubit", wires=self.wires)

    def _circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        @qml.qnode(self.dev, interface="autograd")
        def circuit():
            # Rotation layer
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=self.wires[i])
                qml.RY(rotation_params[3 * i + 1], wires=self.wires[i])
                qml.RZ(rotation_params[3 * i + 2], wires=self.wires[i])
            # Entanglement layer
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[self.wires[i], self.wires[i + 1]])
            # Measure in Z basis
            return [qml.expval(qml.PauliZ(w)) for w in self.wires]
        return circuit

    def run(self, rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024):
        """
        Execute the variational circuit and return a probability
        vector that can be interpreted as attention scores.
        """
        circuit = self._circuit(rotation_params, entangle_params)
        raw = circuit()
        probs = np.exp(raw) / np.sum(np.exp(raw))
        return probs

__all__ = ["SelfAttention"]
