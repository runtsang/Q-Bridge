"""Quantum self‑attention block with expectation‑value evaluation.

The implementation extends the original SelfAttention.py quantum circuit
by integrating the FastBaseEstimator pattern for expectation value
evaluation.  The class can produce either raw measurement counts or
expectation values of user‑supplied Pauli observables.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, List, Sequence

class SelfAttentionHybrid:
    """Quantum self‑attention with expectation‑value evaluation."""

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables : iterable of BaseOperator
            Observables for which expectation values are requested.
        parameter_sets : sequence of parameter sequences
            Each sequence contains concatenated rotation and entangle
            parameters for a single evaluation.

        Returns
        -------
        List[List[complex]]
            Rows correspond to parameter sets, columns to observables.
        """
        results: List[List[complex]] = []
        for values in parameter_sets:
            # Build and bind the circuit
            rotation_len = self.n_qubits * 3
            rot = np.array(values[:rotation_len])
            ent = np.array(values[rotation_len:])
            circuit = self._build_circuit(rot, ent)
            state = Statevector.from_instruction(circuit)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
        observables: Iterable[BaseOperator] | None = None,
    ):
        """Execute the circuit and return either counts or expectation values.

        Parameters
        ----------
        backend : qiskit.providers.Backend
            Target backend for execution.
        rotation_params : np.ndarray
            Rotation angles for each qubit.
        entangle_params : np.ndarray
            Entangling angles between adjacent qubits.
        shots : int
            Number of shots for measurement (ignored when observables are supplied).
        observables : iterable of BaseOperator, optional
            If provided, expectation values are computed on the final statevector;
            otherwise raw counts are returned.

        Returns
        -------
        dict or List[List[complex]]
            If observables are supplied, a list of expectation values per
            observable.  Otherwise the raw measurement counts dictionary.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        if observables is None:
            job = qiskit.execute(circuit, backend, shots=shots)
            return job.result().get_counts(circuit)
        state = Statevector.from_instruction(circuit)
        return [state.expectation_value(obs) for obs in observables]
