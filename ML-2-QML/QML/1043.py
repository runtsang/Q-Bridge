import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector

class SelfAttentionModule:
    """
    Quantum self‑attention block that outputs a weighted sum of value
    vectors using a distribution produced by a variational circuit.
    """

    def __init__(self, seq_len: int, n_qbits: int = None):
        """
        Parameters
        ----------
        seq_len : int
            Length of the input sequence.
        n_qbits : int, optional
            Number of qubits used in the circuit.  If None, defaults to
            ``seq_len``.
        """
        self.seq_len = seq_len
        self.n_qbits = n_qbits or seq_len
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray,
                       entangle_params: np.ndarray):
        """
        Build a variational circuit that yields a probability
        distribution over the sequence positions.
        """
        qc = QuantumCircuit(self.n_qbits)
        # Apply rotations
        for i in range(self.n_qbits):
            params = rotation_params[3 * i: 3 * i + 3]
            qc.rx(params[0], i)
            qc.ry(params[1], i)
            qc.rz(params[2], i)
        # Entangling gates
        for i in range(self.n_qbits - 1):
            qc.cx(i, i + 1)
            qc.rz(entangle_params[i], i)
        # Measure all qubits
        qc.measure_all()
        return qc

    def run(self, inputs: np.ndarray,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024):
        """
        Compute the attention‑weighted sum of the input values.

        Parameters
        ----------
        inputs : np.ndarray
            Shape (batch, seq_len, embed_dim)
        rotation_params : np.ndarray
            Shape (seq_len, 3)
        entangle_params : np.ndarray
            Shape (seq_len-1,)
        shots : int, optional
            Number of shots for the simulator.

        Returns
        -------
        output : np.ndarray
            Shape (batch, seq_len, embed_dim)
        """
        batch, seq_len, embed_dim = inputs.shape
        assert seq_len == self.seq_len, "Sequence length mismatch."

        # Build the circuit once (same for all batch elements)
        qc = self._build_circuit(rotation_params, entangle_params)
        job = execute(qc, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)

        # Convert counts to a probability distribution over positions
        prob_vec = np.zeros(seq_len)
        for bitstring, count in counts.items():
            idx = int(bitstring[::-1], 2) % seq_len
            prob_vec[idx] += count
        prob_vec = prob_vec / prob_vec.sum()

        # Compute weighted sum of values for each batch element
        output = np.empty_like(inputs)
        for b in range(batch):
            # Repeat the distribution across embed_dim
            weights = prob_vec[:, np.newaxis]  # (seq_len, 1)
            output[b] = np.sum(weights * inputs[b], axis=0, keepdims=True)
        return output.squeeze(1)

__all__ = ["SelfAttentionModule"]
