import numpy as np
import pennylane as qml

class QuantumSelfAttention:
    """Variational circuit that emulates a self‑attention block."""

    def __init__(self, n_qubits: int = 4, wires=None, device=None):
        self.n_qubits = n_qubits
        wires = wires or list(range(n_qubits))
        self.device = device or qml.device("default.qubit", wires=wires)
        self.wires = wires

        # Parameter shapes
        self.rotation_shape = (3 * n_qubits,)
        self.entangle_shape = (n_qubits - 1,)

        # Create a QNode
        self.qnode = qml.QNode(self._circuit, self.device)

    def _circuit(self, rotation_params, entangle_params):
        """Variational circuit with Ry, Rz, Rx rotations and CZ entanglement."""
        # Apply single‑qubit rotations
        for i in range(self.n_qubits):
            qml.RY(rotation_params[3 * i], wires=i)
            qml.RZ(rotation_params[3 * i + 1], wires=i)
            qml.RX(rotation_params[3 * i + 2], wires=i)

        # Entanglement layer
        for i in range(self.n_qubits - 1):
            qml.CZ(wires=[i, i + 1])

        # Optional: measurement of all qubits in Z basis
        return [qml.expval(qml.PauliZ(i)) for i in self.wires]

    def run(self,
            backend,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024):
        """
        Execute the circuit and return expectation values as a NumPy array.
        The ``entangle_params`` are unused but accepted for API compatibility.
        """
        if shots is not None:
            self.device.shots = shots
        return self.qnode(rotation_params, entangle_params)

def SelfAttention():
    """Factory returning a ready‑to‑use ``QuantumSelfAttention`` instance."""
    return QuantumSelfAttention(n_qubits=4)

__all__ = ["SelfAttention"]
