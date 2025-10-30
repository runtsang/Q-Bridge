import pennylane as qml
import numpy as np

class SelfAttentionGen064:
    """
    Quantum self‑attention that treats each token as a state prepared by
    parameterised rotations and entanglement.  The resulting feature
    vectors are fed into a simple Gaussian quantum kernel, which is then
    turned into attention scores.
    """
    def __init__(
        self,
        n_qubits: int = 4,
        device: str = "default.qubit",
        shots: int = 1024,
    ):
        self.n_qubits = n_qubits
        self.device = qml.device(device, wires=n_qubits, shots=shots)
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.device, interface="auto")
        def circuit(inputs, rotation_params, entangle_params):
            # Encode the input as rotations (scaled by rotation_params)
            for i in range(self.n_qubits):
                qml.RX(inputs[i] * rotation_params[3 * i], wires=i)
                qml.RY(inputs[i] * rotation_params[3 * i + 1], wires=i)
                qml.RZ(inputs[i] * rotation_params[3 * i + 2], wires=i)

            # Entangle neighbouring qubits
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])

            # Return local Pauli‑Z expectation values as a feature vector
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Flat array of length ``3 * n_qubits`` controlling the input‑dependent
            rotations.
        entangle_params : np.ndarray
            Flat array of length ``n_qubits - 1`` for the controlled‑RX gates.
        inputs : np.ndarray
            Input tensor of shape ``(batch, seq_len, embed_dim)``.  The
            ``embed_dim`` must equal ``n_qubits``.
        Returns
        -------
        np.ndarray
            Output of the quantum attention block, shape ``(batch, seq_len,
            embed_dim)``.
        """
        batch, seq_len, embed_dim = inputs.shape
        if embed_dim!= self.n_qubits:
            raise ValueError("embed_dim must match the number of qubits")

        # Feature extraction: evaluate the circuit for every token
        features = np.zeros((batch, seq_len, self.n_qubits))
        for b in range(batch):
            for s in range(seq_len):
                features[b, s] = self.circuit(inputs[b, s], rotation_params, entangle_params)

        # Quantum‑kernel attention (Gaussian kernel over the feature vectors)
        scores = np.zeros((batch, seq_len, seq_len))
        for b in range(batch):
            for i in range(seq_len):
                for j in range(seq_len):
                    diff = features[b, i] - features[b, j]
                    scores[b, i, j] = np.exp(-0.5 * np.dot(diff, diff))
            # Softmax over the last axis
            scores[b] = np.exp(scores[b]) / np.exp(scores[b]).sum(axis=-1, keepdims=True)

        # Weighted sum of the value vectors (identical to the inputs)
        out = np.einsum("bij,bjk->bik", scores, inputs)
        return out

__all__ = ["SelfAttentionGen064"]
