"""Quantum hybrid self‑attention module using a parameterized overlap circuit.

This implementation encodes two input vectors as quantum states via Ry rotations,
then evaluates their overlap through a simple swap‑test‑inspired circuit.  The
returned similarity matrix can be used as attention weights in a hybrid
classical/quantum attention scheme.

The circuit is intentionally lightweight and runs on the Aer qasm_simulator, but
any backend supporting shot‑based measurement can be supplied.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute

class SelfAttentionHybrid:
    """
    Quantum self‑attention using a parameterized overlap circuit.

    Parameters
    ----------
    n_qubits : int, default 4
        Number of qubits used to encode each input vector.  The input vectors
        must have length equal to ``n_qubits``.
    """

    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_overlap_circuit(self, vec1: np.ndarray, vec2: np.ndarray) -> QuantumCircuit:
        """
        Build a circuit that prepares |psi(vec1)> and then applies the inverse of
        |psi(vec2)>.  The probability of measuring all zeros corresponds to the
        fidelity |<psi(vec1)|psi(vec2)>|^2.
        """
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        # Encode vec1
        for i, val in enumerate(vec1):
            qc.ry(val, i)
        # Apply inverse of vec2
        for i, val in enumerate(vec2):
            qc.ry(-val, i)
        # Measure all qubits
        qc.measure(range(self.n_qubits), range(self.n_qubits))
        return qc

    def run(
        self,
        inputs1: np.ndarray,
        inputs2: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Compute the similarity matrix between two sets of vectors.

        Parameters
        ----------
        inputs1 : np.ndarray of shape (m, n_qubits)
        inputs2 : np.ndarray of shape (n, n_qubits)
        shots : int, optional
            Number of shots for the simulator.

        Returns
        -------
        similarity : np.ndarray of shape (m, n)
            Matrix of overlap values in [0, 1] approximating |<psi1|psi2>|^2.
        """
        m, _ = inputs1.shape
        n, _ = inputs2.shape
        similarity = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                qc = self._build_overlap_circuit(inputs1[i], inputs2[j])
                job = execute(qc, self.backend, shots=shots)
                counts = job.result().get_counts(qc)
                prob_zero = counts.get("0" * self.n_qubits, 0) / shots
                similarity[i, j] = np.sqrt(prob_zero)  # fidelity sqrt
        return similarity

__all__ = ["SelfAttentionHybrid"]
