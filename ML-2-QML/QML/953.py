import pennylane as qml
import numpy as np
from typing import List, Optional

class SelfAttention:
    """
    Quantum self‑attention module implemented with a parameter‑shift variational circuit.
    The run method returns expectation values that can be interpreted as attention scores.
    """
    def __init__(self, n_qubits: int = 4, wires: Optional[List[int]] = None):
        self.n_qubits = n_qubits
        self.wires = wires if wires is not None else list(range(n_qubits))
        self.dev = qml.device("default.qubit", wires=self.wires)

        @qml.qnode(self.dev)
        def circuit(rotation_params, entangle_params):
            # Rotation layer
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=self.wires[i])
                qml.RY(rotation_params[3 * i + 1], wires=self.wires[i])
                qml.RZ(rotation_params[3 * i + 2], wires=self.wires[i])
            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[self.wires[i], self.wires[i + 1]])
                qml.RZ(entangle_params[i], wires=self.wires[i + 1])
            # Measurement
            return [qml.expval(qml.PauliZ(w)) for w in self.wires]

        self.circuit = circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Execute the variational circuit and return expectation values as attention scores.
        """
        rotation_params = np.array(rotation_params, dtype=np.float32)
        entangle_params = np.array(entangle_params, dtype=np.float32)
        expvals = self.circuit(rotation_params, entangle_params)
        return np.array(expvals)

__all__ = ["SelfAttention"]
