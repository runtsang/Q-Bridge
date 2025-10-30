import pennylane as qml
import numpy as np

class SelfAttentionLayer:
    """
    Quantum self‑attention using a variational circuit.
    The circuit outputs a probability distribution over the sequence, used as
    attention weights. The interface mirrors the classical `run` method for
    easy substitution.
    """
    def __init__(self, n_qubits: int = 4, num_layers: int = 2):
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def _circuit(self, rotation_params, entangle_params):
        for layer in range(self.num_layers):
            for q in range(self.n_qubits):
                qml.RX(rotation_params[layer, q, 0], wires=q)
                qml.RY(rotation_params[layer, q, 1], wires=q)
                qml.RZ(rotation_params[layer, q, 2], wires=q)
        for i in range(self.n_qubits - 1):
            qml.CRX(entangle_params[i], wires=[i, i + 1])
        return [qml.probs(wires=i)[1] for i in range(self.n_qubits)]  # prob(|1⟩)

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            inputs: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray, shape (num_layers, n_qubits, 3)
        entangle_params : np.ndarray, shape (n_qubits-1,)
        inputs : np.ndarray, shape (batch, seq_len, embed_dim)
        shots : int

        Returns
        -------
        out : np.ndarray
            Weighted sum of inputs using quantum‑derived attention weights.
        """
        @qml.qnode(self.dev, interface="numpy")
        def circuit():
            return self._circuit(rotation_params, entangle_params)

        probs = circuit()  # shape (n_qubits,)
        attn = probs / probs.sum()  # normalize to sum to 1

        # Broadcast attention weights across batch and embed_dim
        batch, seq_len, embed_dim = inputs.shape
        if seq_len!= self.n_qubits:
            raise ValueError("Input sequence length must match number of qubits")
        attn_tensor = attn.reshape(1, self.n_qubits, 1)
        weighted = inputs * attn_tensor
        out = weighted.sum(axis=1, keepdims=True)  # shape (batch, 1, embed_dim)
        return out.numpy()

    def __repr__(self):
        return f"{self.__class__.__name__}(n_qubits={self.n_qubits}, num_layers={self.num_layers})"

__all__ = ["SelfAttentionLayer"]
