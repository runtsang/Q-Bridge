"""Quantum self‑attention: a variational circuit that outputs attention logits.

The circuit encodes a sequence of classical tokens into a register of
qubits, applies a parameterised rotation layer, entangles neighbouring qubits,
and finally measures in the Z basis.  The resulting expectation values
are interpreted as unnormalised attention scores.  The class exposes a
`run` method compatible with the classical API; it returns a probability
distribution over the sequence positions that can be used directly in a
hybrid loss function.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Tuple, Iterable

class QuantumSelfAttention:
    """
    Variational self‑attention circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits; one qubit per token in the sequence.
    depth : int, optional
        Number of alternating rotation/entanglement layers.
    """

    def __init__(self, num_qubits: int, depth: int = 2) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.encoding = ParameterVector("x", num_qubits)
        self.weights = ParameterVector("theta", num_qubits * depth)
        self.circuit = self._build_circuit()
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 1024

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        # Encoding layer
        for i, param in enumerate(self.encoding):
            qc.rx(param, i)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(self.weights[idx], qubit)
                idx += 1
            # Entangling pattern: nearest‑neighbour CZ
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        # Measurement in Z basis
        qc.measure_all()
        return qc

    def run(
        self,
        token_values: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> np.ndarray:
        """
        Execute the circuit for a single sequence.

        Parameters
        ----------
        token_values : np.ndarray
            1‑D array of token encodings (length == num_qubits).
        rotation_params : np.ndarray
            Parameter vector for the encoding layer (length == num_qubits).
        entangle_params : np.ndarray
            Parameter vector for the variational layers (length == num_qubits * depth).

        Returns
        -------
        np.ndarray
            Normalised attention probability vector over the sequence positions.
        """
        # Bind parameters
        bind_dict = {
            **{str(p): val for p, val in zip(self.encoding, rotation_params)},
            **{str(p): val for p, val in zip(self.weights, entangle_params)},
        }
        bound_qc = self.circuit.bind_parameters(bind_dict)

        job = execute(bound_qc, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(bound_qc)

        # Compute expectation of Z for each qubit: (1 - 2*prob(|1>))
        exp_z = np.zeros(self.num_qubits)
        for bitstring, c in counts.items():
            for i, bit in enumerate(reversed(bitstring)):
                exp_z[i] += (1 - 2 * int(bit)) * c
        exp_z /= self.shots

        # Convert to positive logits and normalise
        logits = np.exp(exp_z - np.max(exp_z))
        return logits / logits.sum()

__all__ = ["QuantumSelfAttention"]
