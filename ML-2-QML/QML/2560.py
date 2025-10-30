"""Hybrid fully connected quantum layer with fast estimator support.

This module provides a parameterised quantum circuit that mimics a fully
connected layer.  It implements a fast estimator for expectation values of
arbitrary `BaseOperator`s, and an optional noisy variant that adds Gaussian
shot noise, mirroring the classical side.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, List, Sequence

class HybridFCL:
    """A parameterised quantum circuit that emulates a fully connected layer."""

    def __init__(self, n_qubits: int = 1, backend=None, shots: int = 100) -> None:
        self._circuit = QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self._parameters = list(self._circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """Return a circuit with parameters bound to the given values."""
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the circuit for each theta and return expectation values."""
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self._circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Fast deterministic estimator for multiple parameter sets and observables.

        Parameters
        ----------
        observables : iterable of BaseOperator
            Each operator whose expectation value is computed.
        parameter_sets : sequence of parameter sequences
            Each inner sequence contains the parameters for one evaluation.

        Returns
        -------
        results : list of lists
            Outer list over parameter sets, inner list over observables.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results

    def evaluate_noisy(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Add Gaussian shot noise to the deterministic estimator.

        Parameters
        ----------
        shots : int, optional
            Number of shots; if None, returns deterministic results.
        seed : int, optional
            Random seed for reproducibility.
        """
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [
                complex(
                    rng.normal(mean.real, max(1e-6, 1 / shots))
                ) + 1j * rng.normal(mean.imag, max(1e-6, 1 / shots))
                for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


def FCL() -> type:
    """Return the HybridFCL class for external usage."""
    return HybridFCL
