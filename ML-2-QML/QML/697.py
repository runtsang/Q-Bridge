"""Quantum self‑attention using a variational circuit.

This module defines a SelfAttention class that builds a parameterised
quantum circuit to emulate an attention block. The circuit encodes the
input vector as rotation angles, applies a layer of single‑qubit
parameterised rotations (rotation_params) and a chain of controlled‑Z
entanglement gates (entangle_params). The resulting probability
distribution is returned as a vector that can be interpreted as
attention scores. The ``run`` method accepts a Qiskit backend, the
parameter arrays, the input data, and the number of shots.

The interface mirrors the original seed, enabling direct substitution
in existing pipelines.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer


class SelfAttention:
    """Variational self‑attention circuit.

    Parameters
    ----------
    n_qubits : int, default 4
        Number of qubits used to represent the attention block.
    """

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> QuantumCircuit:
        """
        Construct a variational circuit that encodes the input data,
        applies parameterised rotations and entanglement, and measures.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameter array of length 3 * n_qubits for RX, RY, RZ gates.
        entangle_params : np.ndarray
            Parameter array of length n_qubits - 1 for CZ entanglement angles.
        inputs : np.ndarray
            Input vector of length n_qubits containing real values to be
            encoded as rotation angles.

        Returns
        -------
        QuantumCircuit
            The constructed circuit ready for execution.
        """
        if rotation_params.size!= 3 * self.n_qubits:
            raise ValueError("rotation_params must have length 3 * n_qubits")
        if entangle_params.size!= self.n_qubits - 1:
            raise ValueError("entangle_params must have length n_qubits - 1")
        if inputs.size!= self.n_qubits:
            raise ValueError("inputs must have length n_qubits")

        circuit = QuantumCircuit(self.qr, self.cr)

        # Encode classical input data into rotations (RY gate)
        for i, val in enumerate(inputs):
            circuit.ry(val, i)

        # Apply parameterised single‑qubit rotations
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        # Entanglement layer with controlled‑Z gates
        for i in range(self.n_qubits - 1):
            circuit.cx(i, i + 1)
            circuit.rz(entangle_params[i], i)
            circuit.cx(i, i + 1)

        # Measure all qubits
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ):
        """
        Execute the variational attention circuit on the provided backend.

        Parameters
        ----------
        backend : qiskit.providers.BaseBackend
            Qiskit backend (simulator or real device).
        rotation_params : np.ndarray
            Parameters for the single‑qubit rotation layer.
        entangle_params : np.ndarray
            Parameters for the entanglement layer.
        inputs : np.ndarray
            Classical input vector to encode.
        shots : int, default 1024
            Number of measurement shots.

        Returns
        -------
        np.ndarray
            Normalised probability vector of length n_qubits derived from
            the measurement counts.
        """
        circuit = self._build_circuit(rotation_params, entangle_params, inputs)
        job = execute(circuit, backend=backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)

        # Convert counts to a probability vector per qubit
        probs = np.zeros(self.n_qubits)
        for state, count in counts.items():
            for i, bit in enumerate(reversed(state)):
                probs[i] += count * int(bit)
        probs /= sum(counts.values())
        return probs

    @staticmethod
    def default_backend():
        """Return a fast local simulator backend."""
        return Aer.get_backend("qasm_simulator")
