import pennylane as qml
import numpy as np

class SelfAttentionModel:
    """
    Quantum self‑attention using a variational circuit.
    The circuit implements learnable single‑qubit rotations followed by
    entangling CNOT layers. The output expectation values of Pauli‑Z
    operators are interpreted as attention scores.
    """

    def __init__(self, n_qubits: int = 4, layers: int = 2):
        """
        Parameters
        ----------
        n_qubits : int
            Number of qubits representing the embedding dimension.
        layers : int
            Number of alternating rotation–entanglement layers.
        """
        self.n_qubits = n_qubits
        self.layers = layers
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=None)

        # Parameter shapes
        self.rotation_shape = (layers, n_qubits, 3)   # RX, RY, RZ per qubit
        self.entangle_shape = (layers, n_qubits - 1)  # CX gates between neighbors

    def _quantum_circuit(self, rotation_params, entangle_params, inputs):
        """
        Build the variational circuit.
        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (layers, n_qubits, 3)
        entangle_params : np.ndarray
            Shape (layers, n_qubits-1)
        inputs : np.ndarray
            Shape (batch, n_qubits) – one-hot or embedded vector.
        Returns
        -------
        qml.QNode
            Executable quantum node.
        """
        @qml.qnode(self.dev)
        def circuit(inputs=inputs):
            # Encode the classical input via rotation on each qubit
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)

            for l in range(self.layers):
                # Rotation layer
                for i in range(self.n_qubits):
                    qml.RX(rotation_params[l, i, 0], wires=i)
                    qml.RY(rotation_params[l, i, 1], wires=i)
                    qml.RZ(rotation_params[l, i, 2], wires=i)

                # Entanglement layer (neighbor CNOTs)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            # Measure expectation values of Pauli‑Z as attention logits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the quantum attention circuit and return a softmaxed
        attention vector over the qubits.
        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (layers, n_qubits, 3)
        entangle_params : np.ndarray
            Shape (layers, n_qubits-1)
        inputs : np.ndarray
            Shape (batch, n_qubits)
        shots : int
            Number of shots for the backend (ignored on default.qubit).
        Returns
        -------
        np.ndarray
            Attention probabilities of shape (batch, n_qubits).
        """
        batch_size = inputs.shape[0]
        out = []
        for b in range(batch_size):
            circuit = self._quantum_circuit(rotation_params, entangle_params, inputs[b])
            logits = circuit()
            # Convert Z expectation (-1 to 1) to logits in [0,1]
            logits = (np.array(logits) + 1) / 2
            probs = np.exp(logits) / np.sum(np.exp(logits))
            out.append(probs)
        return np.stack(out)

__all__ = ["SelfAttentionModel"]
