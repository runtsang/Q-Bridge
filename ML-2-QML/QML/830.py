"""Quantum self‑attention implementation using Qiskit.

This version expands the original 4‑qubit circuit to support an
arbitrary number of qubits, parameterised entanglement, and a
measurement‑based attention distribution.  The public ``run`` API
matches the classical counterpart so that the two modules can be
swapped out in a pipeline.

Example
-------
>>> from qiskit import Aer
>>> backend = Aer.get_backend('qasm_simulator')
>>> sa = SelfAttention(n_qubits=8)
>>> counts = sa.run(backend,
...                 rotation_params=np.random.rand(8,3),
...                 entangle_params=np.random.rand(7),
...                 shots=2048)
>>> sorted(counts.items())[:5]
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute


class SelfAttention:
    """Variational self‑attention circuit."""
    def __init__(self, n_qubits: int):
        if n_qubits < 2:
            raise ValueError("n_qubits must be >= 2")
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        """
        Build a parameterised circuit that mimics the attention pattern.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (n_qubits, 3).  Each row contains the angles for
            RX, RY, RZ applied to the corresponding qubit.
        entangle_params : np.ndarray
            Shape (n_qubits-1,).  Each element is the angle for a
            controlled‑RZ between qubit i and i+1.

        Returns
        -------
        QuantumCircuit
            The fully constructed circuit.
        """
        circuit = QuantumCircuit(self.qr, self.cr)

        # Apply single‑qubit rotations
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[i, 0], i)
            circuit.ry(rotation_params[i, 1], i)
            circuit.rz(rotation_params[i, 2], i)

        # Entangle neighbouring qubits
        for i in range(self.n_qubits - 1):
            circuit.crz(entangle_params[i], i, i + 1)

        # Optional: add a chain of CNOTs to spread correlations
        for i in range(self.n_qubits - 1):
            circuit.cx(i, i + 1)

        # Measure all qubits
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ):
        """
        Execute the circuit on the supplied backend.

        Parameters
        ----------
        backend : qiskit.providers.Backend
            The Qiskit backend to run the circuit on.
        rotation_params : np.ndarray
            Rotation angles for each qubit.
        entangle_params : np.ndarray
            Entanglement angles between adjacent qubits.
        shots : int, optional
            Number of shots for sampling.

        Returns
        -------
        dict
            The measurement counts that can be interpreted as an
            attention probability distribution over the qubits.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

__all__ = ["SelfAttention"]
