import pennylane as qml
import numpy as np

class SelfAttentionModel:
    """
    Quantum self‑attention with a variational circuit.

    The circuit applies a parameterised rotation layer followed by a
    controlled‑entangling layer and additional RZ gates driven by the
    entangle_params.  The expectation values of Pauli‑Z on each qubit are
    mapped to a probability distribution that weights the input embeddings.
    """

    def __init__(self, n_qubits: int, embed_dim: int):
        self.n_qubits = n_qubits
        self.embed_dim = embed_dim
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(rot_params, ent_params, inputs):
            # Rotation layer
            for i in range(n_qubits):
                qml.RX(rot_params[3 * i], wires=i)
                qml.RY(rot_params[3 * i + 1], wires=i)
                qml.RZ(rot_params[3 * i + 2], wires=i)

            # Entanglement layer
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # Additional rotations controlled by entangle_params
            for i in range(n_qubits):
                qml.RZ(ent_params[i], wires=i)

            # Expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def _attention_weights(self, rot_params, ent_params, inputs):
        """Compute attention probabilities from the circuit."""
        probs = self.circuit(rot_params, ent_params, inputs)
        probs = np.array(probs)
        probs = (probs + 1) / 2  # map from [-1,1] to [0,1]
        probs = probs / probs.sum()
        return probs

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        """
        Return a weighted sum of ``inputs`` using quantum attention.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the rotation layer (length 3 * n_qubits).
        entangle_params : np.ndarray
            Parameters for the RZ gates after entanglement (length n_qubits).
        inputs : np.ndarray
            Shape (seq_len, embed_dim).  ``seq_len`` must equal ``n_qubits``.
        shots : int, optional
            Number of shots for sampling (ignored in the default autograd device).
        """
        seq_len, embed_dim = inputs.shape
        assert seq_len == self.n_qubits, "Number of qubits must equal sequence length"

        probs = self._attention_weights(rotation_params, entangle_params, inputs)
        weighted = probs[:, None] * inputs
        return weighted.sum(axis=0)
