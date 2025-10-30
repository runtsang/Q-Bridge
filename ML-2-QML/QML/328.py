import pennylane as qml
import numpy as np

class SelfAttention__gen364:
    """
    Variational quantum self‑attention block.  Uses a PennyLane device
    to encode input embeddings into a quantum state, applies a
    parameterized circuit that mimics the QK^T scaling, and measures
    expectation values to produce attention scores.
    """

    def __init__(self, embed_dim: int, n_qubits: int = 6, device_name: str = "default.qubit"):
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.dev = qml.device(device_name, wires=n_qubits)
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev)
        def circuit(rparams, eparams, inputs):
            # Encode inputs via rotations
            for i in range(self.n_qubits):
                qml.RX(inputs[i], wires=i)
            # Rotate with query params
            for i in range(self.n_qubits):
                qml.RY(rparams[i], wires=i)
            # Entangle with key params
            for i in range(self.n_qubits - 1):
                qml.CRX(eparams[i], wires=[i, i + 1])
            # Measure expectation of PauliZ for each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Executes the variational attention circuit.
        Parameters
        ----------
        rotation_params : array shape (n_qubits,)
            Parameters for the rotation (query) stage.
        entangle_params : array shape (n_qubits-1,)
            Parameters for the entanglement (key) stage.
        inputs : array shape (n_qubits,)
            Input embedding vector to be encoded.
        shots : int
            Number of shots for the simulation.

        Returns
        -------
        attention_matrix : array shape (n_qubits, n_qubits)
            Approximate attention scores derived from measurement outcomes.
        """
        rotation_params = np.asarray(rotation_params).reshape(-1)
        entangle_params = np.asarray(entangle_params).reshape(-1)
        inputs = np.asarray(inputs).reshape(-1)

        result = self.circuit(rotation_params, entangle_params, inputs)
        probs = (np.array(result) + 1) / 2  # map [-1,1] to [0,1]
        attention_matrix = np.outer(probs, probs)
        attention_matrix /= attention_matrix.sum(axis=-1, keepdims=True) + 1e-8
        return attention_matrix

__all__ = ["SelfAttention__gen364"]
