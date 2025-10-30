import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

class QuantumSelfAttentionPenny:
    """
    Variational quantum circuit that mimics a self‑attention block.
    rotation_params control single‑qubit rotations; entangle_params
    control controlled‑RX gates between neighboring qubits.
    """
    def __init__(self, n_qubits: int = 4, device: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=n_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        for i in range(self.n_qubits):
            qml.RX(rotation_params[3 * i], wires=i)
            qml.RY(rotation_params[3 * i + 1], wires=i)
            qml.RZ(rotation_params[3 * i + 2], wires=i)
        for i in range(self.n_qubits - 1):
            qml.CRX(entangle_params[i], wires=[i, i + 1])
        return qml.state()

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024):
        """
        Execute the circuit and return a probability distribution over basis states.
        """
        state = self.qnode(rotation_params, entangle_params)
        probs = np.abs(state)**2
        probs = probs / probs.sum()
        return probs

def SelfAttention():
    return QuantumSelfAttentionPenny(n_qubits=4)

__all__ = ["SelfAttention"]
