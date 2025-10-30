"""Quantum self‑attention using Qiskit with parameterized ansatz and entanglement."""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute


class SelfAttention:
    """
    Quantum self‑attention block.
    Parameters are used to build a parameterized circuit that outputs a probability
    distribution over qubits; this distribution is used as attention weights.
    """

    def __init__(self, n_qubits: int, backend=None):
        """
        Args:
            n_qubits: Number of qubits (must match embed_dim).
            backend: Qiskit backend; if None, a local simulator is used.
        """
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = backend or Aer.get_backend("qasm_simulator")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        """
        Build a parameterized circuit.

        rotation_params: shape (n_qubits, 3) – angles for RX, RY, RZ on each qubit.
        entangle_params: shape (n_qubits-1,) – angles for CRX gates between consecutive qubits.
        """
        qc = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[i, 0], i)
            qc.ry(rotation_params[i, 1], i)
            qc.rz(rotation_params[i, 2], i)

        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)

        qc.measure(self.qr, self.cr)
        return qc

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the circuit and use the measurement distribution as attention weights.

        Parameters
        ----------
        rotation_params: shape (n_qubits, 3)
        entangle_params: shape (n_qubits-1,)
        inputs: shape (n_qubits,) – the embeddings to be weighted.
        shots: number of shots for the simulator.

        Returns
        -------
        weighted_output: shape (n_qubits,) – weighted sum of inputs based on quantum
                         measurement probabilities.
        """
        qc = self._build_circuit(rotation_params, entangle_params)
        job = execute(qc, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)

        # Compute expectation value of Pauli‑Z for each qubit
        exp_vals = np.zeros(self.n_qubits)
        for bitstring, cnt in counts.items():
            prob = cnt / shots
            # bitstring is in little‑endian order
            for j in range(self.n_qubits):
                if bitstring[self.n_qubits - 1 - j] == "0":
                    exp_vals[j] += prob
                else:
                    exp_vals[j] -= prob

        # Convert to [0, 1] weights
        weights = (exp_vals + 1) / 2

        weighted_output = weights * inputs
        return weighted_output


__all__ = ["SelfAttention"]
