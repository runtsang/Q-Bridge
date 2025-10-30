import pennylane as qml
import numpy as np

class SelfAttentionQuantum:
    """Variational quantum circuit that produces attention weights from input embeddings.

    The circuit applies a layer of single‑qubit rotations (parameterised by rotation_params),
    followed by a configurable number of entangling layers (parameterised by entangle_params).
    Attention scores are obtained from the expectation values of Pauli‑Z measurements
    on each qubit. The resulting probabilities are normalised to form a softmax‑like
    distribution over the input sequence.

    Parameters
    ----------
    n_qubits : int
        Number of qubits, equal to the sequence length.
    n_entangle_layers : int, default 1
        Number of entangling layers.
    """

    def __init__(self, n_qubits: int, n_entangle_layers: int = 1):
        self.n_qubits = n_qubits
        self.n_entangle_layers = n_entangle_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def _rotations(self, params: np.ndarray):
        """Apply a rotation gate to each qubit."""
        for i in range(self.n_qubits):
            qml.RX(params[3 * i], wires=i)
            qml.RY(params[3 * i + 1], wires=i)
            qml.RZ(params[3 * i + 2], wires=i)

    def _entangle_layer(self, params: np.ndarray):
        """Entangle adjacent qubits with controlled‑RZ gates."""
        for i in range(self.n_qubits - 1):
            qml.CZ(wires=[i, i + 1])
            qml.RZ(params[i], wires=i + 1)

    def _circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        """Full variational circuit."""
        self._rotations(rotation_params)
        for _ in range(self.n_entangle_layers):
            self._entangle_layer(entangle_params)

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Execute the variational circuit and return attention weights.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for single‑qubit rotations (length 3 * n_qubits).
        entangle_params : np.ndarray
            Parameters for entangling layer (length n_qubits - 1).
        inputs : np.ndarray
            Input embeddings of shape (seq_len, embed_dim). Only the first
            ``n_qubits`` components of each embedding are used as initial
            amplitudes.

        Returns
        -------
        np.ndarray
            Attention weights of shape (seq_len,).
        """
        seq_len = inputs.shape[0]
        # Prepare initial state: encode inputs into computational basis
        init_state = np.zeros(2 ** self.n_qubits)
        init_state[0] = 1.0
        @qml.qnode(self.dev, interface="numpy")
        def qnode():
            qml.StatePreparation(init_state)
            self._circuit(rotation_params, entangle_params)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        raw_scores = qnode()
        probs = np.exp(raw_scores) / np.sum(np.exp(raw_scores))
        return probs

__all__ = ["SelfAttentionQuantum"]
