import math
import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute

class SelfAttention:
    """
    Purely quantum self‑attention module that implements the attention
    mechanism with a variational circuit.  For each query‑key pair a
    single‑qubit circuit is executed; the probability of measuring |0>
    is interpreted as the attention weight.  The design is inspired by
    the QCNN ansatz used in the reference QCNN code, where the qubit
    rotations encode the similarity between query and key and
    entangling gates provide a richer feature space.
    """

    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

    def _build_circuit(self, angle: float) -> QuantumCircuit:
        qc = QuantumCircuit(1, 1)
        qc.rx(angle, 0)
        qc.measure(0, 0)
        return qc

    def run(self, inputs: np.ndarray,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray) -> np.ndarray:
        """
        Execute the quantum self‑attention for a batch of sequences.

        Parameters
        ----------
        inputs : np.ndarray
            Shape (batch, seq_len, embed_dim).
        rotation_params, entangle_params : np.ndarray
            Parameters for the quantum circuit (kept for API
            compatibility; not used in this pure‑quantum implementation).

        Returns
        -------
        np.ndarray
            Attention weight matrix of shape (batch, seq_len, seq_len).
        """
        batch, seq_len, _ = inputs.shape
        weights = np.zeros((batch, seq_len, seq_len))
        for b in range(batch):
            for i in range(seq_len):
                query = inputs[b, i]
                for j in range(seq_len):
                    key = inputs[b, j]
                    angle = float(np.dot(query, key)) * math.pi
                    qc = self._build_circuit(angle)
                    job = execute(qc, self.backend, shots=self.shots)
                    counts = job.result().get_counts()
                    prob_zero = counts.get('0', 0) / self.shots
                    weights[b, i, j] = prob_zero
            # normalise across keys
            weights[b] = weights[b] / weights[b].sum(axis=-1, keepdims=True)
        return weights

__all__ = ["SelfAttention"]
