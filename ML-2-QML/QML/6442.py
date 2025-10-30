import pennylane as qml
import numpy as np

class SelfAttentionHybrid:
    """
    Quantum self‑attention block implemented with PennyLane.
    Uses a variational circuit that mirrors the classical attention
    pattern: rotation parameters for each qubit and entanglement
    parameters for nearest‑neighbour CNOT‑like gates.
    """

    def __init__(self, n_qubits: int = 4, dev_name: str = "default.qubit", shots: int = 1024):
        self.n_qubits = n_qubits
        self.dev = qml.device(dev_name, wires=n_qubits, shots=shots)

    def _circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        @qml.qnode(self.dev, interface="autograd")
        def circuit():
            # Apply single‑qubit rotations
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)
            # Entangle neighboring qubits
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                # Optional parameterized entangling gate
                qml.RZ(entangle_params[i], wires=i + 1)
            # Measure expectation of Z on each wire
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> np.ndarray:
        """
        Execute the variational circuit and return the expectation values
        as a proxy for attention weights.  The outputs can be post‑processed
        to form a probability distribution.
        """
        circuit = self._circuit(rotation_params, entangle_params)
        return circuit()

__all__ = ["SelfAttentionHybrid"]
