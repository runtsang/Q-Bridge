"""Quantum self‑attention built with a parameter‑shaped variational circuit.

The quantum implementation mirrors the classical API.  It accepts
``rotation_params`` and ``entangle_params`` to parameterise a
variational circuit.  The circuit consists of single‑qubit rotations
followed by a configurable depth of entangling CRX gates.  After
measurement the outcome frequencies are returned as a dictionary
mapping bitstrings to counts.  The ``run`` method accepts a Qiskit
backend, the parameters, and the number of shots.

Usage
-----
>>> from qiskit import Aer
>>> backend = Aer.get_backend('qasm_simulator')
>>> sa = SelfAttention(n_qubits=4, entanglement_depth=2)
>>> out = sa.run(backend, rotation_params, entangle_params, shots=1024)
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute

class SelfAttention:
    """Variational quantum self‑attention circuit."""

    def __init__(self, n_qubits: int, entanglement_depth: int = 1):
        self.n_qubits = n_qubits
        self.entanglement_depth = entanglement_depth
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self,
                       rotation_params: np.ndarray,
                       entangle_params: np.ndarray) -> QuantumCircuit:
        """
        Build a variational circuit with the given parameters.

        Parameters
        ----------
        rotation_params : np.ndarray
            Length 3 * n_qubits; parameters for RX, RY, RZ on each qubit.
        entangle_params : np.ndarray
            Length (n_qubits - 1) * entanglement_depth for CRX gates between
            adjacent qubits.  The pattern repeats for each depth layer.
        """
        circuit = QuantumCircuit(self.qr, self.cr)

        # Apply single‑qubit rotations
        for i in range(self.n_qubits):
            idx = 3 * i
            circuit.rx(rotation_params[idx], i)
            circuit.ry(rotation_params[idx + 1], i)
            circuit.rz(rotation_params[idx + 2], i)

        # Entangling layers
        for d in range(self.entanglement_depth):
            for i in range(self.n_qubits - 1):
                param_idx = d * (self.n_qubits - 1) + i
                circuit.crx(entangle_params[param_idx], i, i + 1)

        # Measurement
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self,
            backend,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024) -> dict:
        """
        Execute the circuit on the given backend.

        Parameters
        ----------
        backend : qiskit.providers.BaseBackend
            The backend to run the circuit on.
        rotation_params, entangle_params : np.ndarray
            Parameters for the variational circuit.
        shots : int
            Number of measurement shots.

        Returns
        -------
        dict
            Bitstring counts from the measurement.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend=backend, shots=shots)
        result = job.result()
        return result.get_counts(circuit)
