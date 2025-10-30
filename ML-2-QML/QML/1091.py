import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

class HybridSelfAttention:
    """
    Variational quantum circuit that emulates a self‑attention block.
    The rotation and entanglement parameters are mapped to single‑qubit
    rotations and two‑qubit CX‑plus‑RZ gates.  The resulting expectation
    values of Pauli‑Z are turned into a probability distribution that
    serves as the attention weight vector.
    """
    def __init__(self, n_qubits: int, embed_dim: int):
        self.n_qubits = n_qubits
        self.embed_dim = embed_dim
        self.qreg = QuantumRegister(n_qubits, "q")
        self.creg = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray,
                       entangle_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qreg, self.creg)

        # Single‑qubit rotations
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)

        # Entangling layer (CX followed by an RZ on the target)
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(entangle_params[i], i + 1)

        return qc

    def run(self, backend, rotation_params: np.ndarray,
            entangle_params: np.ndarray, inputs: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        """
        Execute the circuit and compute an attention‑weighted representation.
        Parameters
        ----------
        backend : qiskit.providers.backend.Backend
            The simulator or real device to use.
        rotation_params : np.ndarray
            Flat array of length 3*n_qubits containing Rx, Ry, Rz angles.
        entangle_params : np.ndarray
            Flat array of length n_qubits-1 containing Rz angles for the
            entangling targets.
        inputs : np.ndarray
            Input features of shape (batch, embed_dim).
        shots : int
            Number of measurement shots.
        Returns
        -------
        np.ndarray
            The attended representation of shape (batch, embed_dim).
        """
        qc = self._build_circuit(rotation_params, entangle_params)

        # Execute the circuit
        job = execute(qc, backend, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)

        # Convert counts to probabilities
        probs = np.array([counts.get(bin(i)[2:].zfill(self.n_qubits), 0)
                          for i in range(2 ** self.n_qubits)]) / shots

        # Expectation value of Pauli‑Z for each qubit
        exp_vals = np.zeros(self.n_qubits)
        for i in range(self.n_qubits):
            exp = 0.0
            for state, prob in zip(counts.keys(), probs):
                bit = int(state[self.n_qubits - 1 - i])  # state string is MSB first
                exp += ((-1) ** bit) * prob
            exp_vals[i] = exp

        # Convert expectation values to a probability distribution
        attn_weights = np.exp(exp_vals) / np.sum(np.exp(exp_vals))

        # Attention‑weighted sum of the inputs
        # Broadcast the weights across the batch dimension
        output = attn_weights.reshape(1, -1) * inputs
        return output

__all__ = ["HybridSelfAttention"]
