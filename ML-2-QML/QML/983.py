"""Quantum self‑attention using Pennylane with parameter‑shift gradients."""
import pennylane as qml
import numpy as np

class SelfAttention:
    """
    Quantum self‑attention block implemented with Pennylane.
    Parameters
    ----------
    n_qubits : int
        Number of qubits representing the embedding dimension.
    entangler_map : list[tuple[int, int]] | None, default=None
        Custom entanglement pattern. Defaults to consecutive CNOTs.
    """
    def __init__(self, n_qubits: int,
                 entangler_map: list[tuple[int, int]] | None = None):
        self.n_qubits = n_qubits
        self.entangler_map = entangler_map or [(i, i + 1) for i in range(n_qubits - 1)]
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self._build_qnode()

    def _rotation_layer(self, params: np.ndarray):
        """Apply a layer of RX, RY, RZ rotations to each qubit."""
        for i in range(self.n_qubits):
            qml.RX(params[3 * i], wires=i)
            qml.RY(params[3 * i + 1], wires=i)
            qml.RZ(params[3 * i + 2], wires=i)

    def _entangling_layer(self, params: np.ndarray):
        """Apply entangling CNOTs with optional parameterized rotations."""
        for idx, (i, j) in enumerate(self.entangler_map):
            qml.CNOT(wires=[i, j])
            qml.RZ(params[idx], wires=j)

    def _qnode(self, rotation_params, entangle_params):
        @qml.qnode(self.dev, interface="autograd")
        def circuit():
            self._rotation_layer(rotation_params)
            self._entangling_layer(entangle_params)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return circuit

    def _build_qnode(self):
        self.qnode = self._qnode

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray) -> np.ndarray:
        """
        Execute the quantum self‑attention circuit.
        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the rotation layer (size 3 * n_qubits).
        entangle_params : np.ndarray
            Parameters for the entangling layer (size len(entangler_map)).
        Returns
        -------
        np.ndarray
            Attention logits interpreted as a probability distribution over qubits.
        """
        circuit = self.qnode(rotation_params, entangle_params)
        logits = circuit()
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        return probs

    def gradient(self,
                 rotation_params: np.ndarray,
                 entangle_params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients of the attention logits w.r.t. the parameters using
        Pennylane's automatic differentiation.
        Returns
        -------
        tuple
            Gradients for rotation_params and entangle_params.
        """
        grad_fn = qml.grad(self.qnode)
        return grad_fn(rotation_params, entangle_params)

__all__ = ["SelfAttention"]
