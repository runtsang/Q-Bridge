"""Quantum self‑attention using a parameterized circuit.

The circuit applies Rx,Ry,Rz rotations per qubit and
controlled‑Rz gates between adjacent qubits.  The
run method executes the circuit on the supplied backend and
returns both measurement counts and a vector of
Pauli‑Z expectation values that can be interpreted as
attention weights.

The interface mirrors the original: run(backend, rotation_params, entangle_params, shots=1024).
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

class SelfAttention__gen191:
    """Quantum self‑attention with variational parameters."""

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        """Construct a parameterized circuit from the supplied arrays."""
        circuit = QuantumCircuit(self.qr, self.cr)
        # Apply single‑qubit rotations
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        # Entangle adjacent qubits with controlled‑Rz
        for i in range(self.n_qubits - 1):
            circuit.cx(i, i + 1)
            circuit.rz(entangle_params[i], i)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        """
        Execute the circuit and return measurement statistics and
        expectation values of Pauli‑Z on each qubit.

        Parameters
        ----------
        backend : qiskit.providers.Provider
            Backend to execute the circuit on.
        rotation_params : np.ndarray
            Shape (3 * n_qubits,).  Rotation angles for each qubit.
        entangle_params : np.ndarray
            Shape (n_qubits - 1,).  Rotation angles for entangling gates.
        shots : int, optional
            Number of measurement shots.

        Returns
        -------
        dict
            {'counts':..., 'attention_weights': np.ndarray}
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)

        # Compute Pauli‑Z expectation values per qubit
        exp_vals = []
        for qubit in range(self.n_qubits):
            # Probability of |0> minus |1>
            p0 = sum(count / shots for bitstring, count in counts.items()
                     if bitstring[self.n_qubits - 1 - qubit] == "0")
            p1 = 1.0 - p0
            exp_vals.append(p0 - p1)
        return {"counts": counts, "attention_weights": np.array(exp_vals)}

# Default backend for convenience
backend = Aer.get_backend("qasm_simulator")

__all__ = ["SelfAttention__gen191"]
