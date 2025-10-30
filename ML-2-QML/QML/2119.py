import pennylane as qml
import numpy as np

class SelfAttentionModel:
    """Variational quantum self‑attention that outputs attention weights
    from Pauli‑Z expectation values. These weights are used to compute
    a weighted sum of the classical input feature vectors."""
    def __init__(self, n_qubits: int = 4, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    def _q_circuit(self, rotation_params, entangle_params, features):
        """
        rotation_params: shape (n_qubits * 3,)
        entangle_params: shape (n_qubits - 1,)
        features: shape (n_qubits,)  # classical feature vector
        """
        for i in range(self.n_qubits):
            qml.RX(rotation_params[3 * i], wires=i)
            qml.RY(rotation_params[3 * i + 1], wires=i)
            qml.RZ(rotation_params[3 * i + 2], wires=i)
            # encode feature via controlled phase shift
            qml.ctrl(qml.PhaseShift, control=features[i], target=i)(np.pi / 2)

        for i in range(self.n_qubits - 1):
            qml.CRX(entangle_params[i], wires=[i, i + 1])

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def _build_qnode(self):
        @qml.qnode(self.dev)
        def qnode(rotation_params, entangle_params, features):
            return self._q_circuit(rotation_params, entangle_params, features)
        return qnode

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            inputs: np.ndarray, backend=None) -> np.ndarray:
        """
        Computes attention weights from the quantum circuit and
        applies them to the classical input feature matrix.
        Args:
            rotation_params: shape (n_qubits * 3,)
            entangle_params: shape (n_qubits - 1,)
            inputs: shape (seq_len, n_qubits)
        Returns:
            weighted sum: shape (seq_len, n_qubits)
        """
        seq_len = inputs.shape[0]
        qnode = self._build_qnode()
        weights = []

        for t in range(seq_len):
            raw = qnode(rotation_params, entangle_params, inputs[t])
            # Map from [-1, 1] to [0, 1] and normalize
            w = (np.array(raw) + 1.0) / 2.0
            w = w / np.sum(w)
            weights.append(w)

        weights = np.stack(weights, axis=0)  # (seq_len, n_qubits)
        weighted = (weights[..., None] * inputs[..., None]).sum(axis=1)
        return weighted

__all__ = ["SelfAttentionModel"]
