"""Quantum self‑attention using PennyLane.

The class implements a parameterised variational circuit that
encodes a 1‑D sequence into qubit rotations, applies rotation and
entanglement layers, and returns a weighted sum of the input
tokens.  The API matches the classical version: ``run(backend,
rotation_params, entangle_params, inputs, shots)``.
"""
import pennylane as qml
import numpy as np

class SelfAttention:
    """
    Quantum self‑attention module.

    Parameters
    ----------
    n_qubits : int
        Number of qubits per token.
    seq_len : int
        Length of the input sequence.
    """
    def __init__(self, n_qubits: int, seq_len: int):
        self.n_qubits = n_qubits
        self.seq_len = seq_len
        self.width = n_qubits * seq_len

    def _angle_encode(self, circuit, x):
        """Encode a 1‑D vector into qubit rotations."""
        for i in range(self.seq_len):
            for q in range(self.n_qubits):
                idx = i * self.n_qubits + q
                circuit += qml.RX(x[i] * np.pi, wires=idx)

    def _variational_layer(self, circuit, rotation_params, entangle_params):
        """Add rotation and entanglement gates."""
        # Single‑qubit rotations
        for i in range(self.width):
            r = rotation_params[i]
            circuit += qml.Rot(r[0], r[1], r[2], wires=i)
        # Entanglement within a token
        for i in range(0, self.width - 1, 2):
            circuit += qml.CNOT(wires=[i, i + 1])
        # Entanglement across tokens
        for i in range(0, self.width - self.n_qubits, self.n_qubits):
            circuit += qml.CNOT(wires=[i, i + self.n_qubits])
        return circuit

    def run(self, backend: str, rotation_params: np.ndarray,
            entangle_params: np.ndarray, inputs: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        """
        Execute the attention circuit on a batch of inputs.

        Parameters
        ----------
        backend : str
            PennyLane device name (e.g., ``"default.qubit"``, ``"qiskit_aer"``).
        rotation_params : np.ndarray
            Shape (width, 3) array of Euler angles for each qubit.
        entangle_params : np.ndarray
            Shape (width,) array of entanglement strengths (ignored in this
            implementation but kept for API compatibility).
        inputs : np.ndarray
            Shape (batch, seq_len) – raw token values.
        shots : int, default 1024
            Number of measurement shots.

        Returns
        -------
        np.ndarray
            Weighted sum of the input embeddings per batch, shape
            (batch,).
        """
        dev = qml.device(backend, wires=self.width, shots=shots)

        @qml.qnode(dev)
        def circuit(x):
            # Angle encoding
            for i in range(self.seq_len):
                for q in range(self.n_qubits):
                    idx = i * self.n_qubits + q
                    qml.RX(x[i] * np.pi, wires=idx)
            # Variational layer
            for i in range(self.width):
                r = rotation_params[i]
                qml.Rot(r[0], r[1], r[2], wires=i)
            for i in range(0, self.width - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            for i in range(0, self.width - self.n_qubits, self.n_qubits):
                qml.CNOT(wires=[i, i + self.n_qubits])
            # Return expectation values of Pauli‑Z
            return [qml.expval(qml.PauliZ(i)) for i in range(self.width)]

        batch_out = []
        for x in inputs:
            probs = circuit(x)
            probs = np.array(probs).reshape(self.seq_len, self.n_qubits)
            probs = (probs + 1) / 2  # map from [-1,1] to [0,1]
            attn_weights = probs.mean(axis=1)
            batch_out.append((attn_weights * x).sum())
        return np.array(batch_out)

__all__ = ["SelfAttention"]
