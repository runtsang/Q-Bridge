import pennylane as qml
import numpy as np

class QuantumSelfAttention:
    """
    Variational quantum circuit that produces an attention matrix
    from rotation and entanglement parameters.
    """

    def __init__(self, num_qubits: int):
        """
        Parameters
        ----------
        num_qubits : int
            Number of qubits (should match the head dimension).
        """
        self.num_qubits = num_qubits
        self.dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(self.dev)
        def circuit(rotation_params, entangle_params):
            # Apply parameterized rotations
            for i in range(self.num_qubits):
                qml.RX(rotation_params[i, 0], wires=i)
                qml.RY(rotation_params[i, 1], wires=i)
                qml.RZ(rotation_params[i, 2], wires=i)
            # Entanglement layer (simple chain)
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measure expectation values of Pauli‑Z
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.circuit = circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> np.ndarray:
        """
        Execute the circuit and return an attention matrix.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (num_qubits, 3) containing RX, RY, RZ parameters.
        entangle_params : np.ndarray
            Unused in this simple implementation but kept for API compatibility.

        Returns
        -------
        np.ndarray
            Attention matrix of shape (num_qubits, num_qubits).
        """
        # Execute the circuit
        z_expect = self.circuit(rotation_params, entangle_params)
        # Convert expectation values to probabilities
        probs = np.clip(np.array(z_expect), -1, 1)
        probs = (probs + 1) / 2  # map to [0, 1]
        # Compute a simple outer‑product attention matrix
        attn = np.outer(probs, probs)
        # Normalize rows to sum to 1
        attn = attn / attn.sum(axis=-1, keepdims=True)
        return attn
