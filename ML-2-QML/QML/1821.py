"""Quantum self‑attention built with PennyLane.

The circuit uses parameterised rotations and controlled‑RX entanglement
to produce a distribution over qubits that is interpreted as
attention weights.  The module follows the same API as the classical
version, retaining rotation_params and entangle_params arguments for
compatibility while computing expectation values of Pauli‑Z
measurements.  Gradients can be obtained via PennyLane’s automatic
differentiation.
"""

import numpy as np
import pennylane as qml

class SelfAttentionGen422:
    """Variational attention circuit implemented with PennyLane."""
    def __init__(self, n_qubits: int, device=None):
        self.n_qubits = n_qubits
        self.device = device or qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(self._circuit, self.device)

    def _circuit(self, rotation_params, entangle_params):
        # rotation_params shape: (n_qubits, 3)
        for i in range(self.n_qubits):
            qml.RX(rotation_params[i, 0], wires=i)
            qml.RY(rotation_params[i, 1], wires=i)
            qml.RZ(rotation_params[i, 2], wires=i)
        for i in range(self.n_qubits - 1):
            qml.CRX(entangle_params[i], wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        # Convert inputs to numpy arrays
        rotation_params = np.array(rotation_params, dtype=float)
        entangle_params = np.array(entangle_params, dtype=float)
        # Expectation values of Z on each qubit
        raw = self.qnode(rotation_params, entangle_params)
        # Map [-1,1] → [0,1], then softmax to obtain a probability vector
        probs = (np.array(raw) + 1) / 2
        exp = np.exp(probs - np.max(probs))
        return exp / exp.sum()

    def gradient(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        """Return gradients of the circuit w.r.t. all parameters."""
        grad_fn = qml.grad(self._circuit, argnum=[0, 1])
        return grad_fn(np.array(rotation_params, dtype=float),
                       np.array(entangle_params, dtype=float))

__all__ = ["SelfAttentionGen422"]
