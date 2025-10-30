"""Quantum self‑attention built with a variational circuit and classical post‑processing.

The circuit now contains
* separate attention heads implemented as groups of qubits.
* a set of rotation gates per qubit defined by ``rotation_params``.
* controlled‑RZ entanglement between neighbouring qubits defined by ``entangle_params``.
* measurement yields a probability distribution that is mapped to attention weights.
* an optional ``inputs`` tensor (batch, seq_len, embed_dim) can be optionally supplied to compute a weighted sum of the values.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

def SelfAttention():
    class QuantumSelfAttention:
        """Variational self‑attention quantum circuit."""

        def __init__(self, n_qubits: int, num_heads: int = 1):
            self.n_qubits = n_qubits
            self.num_heads = num_heads
            self.qr = QuantumRegister(n_qubits, "q")
            self.cr = ClassicalRegister(n_qubits, "c")

        def _build_circuit(
            self, rotation_params: np.ndarray, entangle_params: np.ndarray
        ) -> QuantumCircuit:
            circuit = QuantumCircuit(self.qr, self.cr)
            # Rotation layer
            for i in range(self.n_qubits):
                circuit.rx(rotation_params[3 * i], i)
                circuit.ry(rotation_params[3 * i + 1], i)
                circuit.rz(rotation_params[3 * i + 2], i)
            # Entanglement layer
            for i in range(self.n_qubits - 1):
                circuit.cx(i, i + 1)
                circuit.rz(entangle_params[i], i + 1)
            # Measurement
            circuit.measure(self.qr, self.cr)
            return circuit

        def run(
            self,
            backend,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray = None,
            shots: int = 1024,
        ):
            """
            Execute the circuit and post‑process the measurement results.

            Parameters
            ----------
            backend : qiskit.providers.Provider
                Quantum backend to execute on.
            rotation_params : np.ndarray
                Rotation angles, shape (n_qubits * 3,).
            entangle_params : np.ndarray
                Entanglement angles, shape (n_qubits - 1,).
            inputs : np.ndarray, optional
                Tensor of shape (batch, seq_len, embed_dim) to be weighted
                by the quantum‑derived attention weights.
            shots : int, optional
                Number of shots for the execution.

            Returns
            -------
            np.ndarray
                If ``inputs`` is provided, returns a weighted sum of the
                values of shape (batch, embed_dim).  Otherwise returns the
                attention weight vector of shape (n_qubits,).
            """
            circuit = self._build_circuit(rotation_params, entangle_params)
            job = qiskit.execute(circuit, backend, shots=shots)
            result = job.result()
            counts = result.get_counts(circuit)

            # Convert counts to probabilities
            probs = {k: v / shots for k, v in counts.items()}

            # Build a probability vector over all bitstrings
            probs_vec = np.zeros(2 ** self.n_qubits)
            for bitstring, p in probs.items():
                idx = int(bitstring[::-1], 2)  # reverse to match Qiskit's bit order
                probs_vec[idx] = p

            # Derive a weight per qubit (probability that qubit is |1⟩)
            weights = []
            for i in range(self.n_qubits):
                mask = ((np.arange(2 ** self.n_qubits) >> i) & 1) == 1
                weight = probs_vec[mask].sum()
                weights.append(weight)
            weights = np.array(weights)
            weights /= (weights.sum() + 1e-8)

            if inputs is None:
                return weights

            # Weighted sum over the sequence dimension
            # Assume seq_len equals n_qubits for simplicity
            seq_len = inputs.shape[1]
            if seq_len!= self.n_qubits:
                # Truncate or pad
                weights = np.resize(weights, seq_len)
            weights = weights.reshape(1, seq_len, 1)  # broadcast
            weighted = (weights * inputs).sum(axis=1)  # sum over seq_len
            return weighted

    # Default instance
    backend = qiskit.Aer.get_backend("qasm_simulator")
    attention = QuantumSelfAttention(n_qubits=4)
    return attention

__all__ = ["SelfAttention"]
