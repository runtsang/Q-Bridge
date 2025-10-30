"""FastBaseEstimator__gen195: Quantum estimator using Qiskit Statevector simulation.

The class accepts a parameterised QuantumCircuit and a list of qiskit
BaseOperator observables.  It can evaluate expectation values in batch and
optionally add shot‑noise.  The implementation mirrors the classical
interface for API compatibility.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Union, Optional, Callable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

ComplexObservable = Callable[[Statevector], complex | float]

class FastBaseEstimator__gen195:
    """Hybrid estimator that evaluates a parameterised quantum circuit.

    Parameters
    ----------
    circuit : qiskit.circuit.QuantumCircuit
        A circuit with symbolic parameters that will be bound for each
        evaluation run.
    """
    def __init__(self, circuit: QuantumCircuit) -> None:
        self.circuit = circuit
        self._param_names = list(circuit.parameters)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self._param_names):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._param_names, param_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        noise: str | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set.

        Parameters
        ----------
        observables : iterable of qiskit.quantum_info.operators.base_operator.BaseOperator
            Observables to evaluate on the statevector.
        parameter_sets : list of sequences
            Each inner sequence contains the numeric parameters for one
            circuit execution.
        shots : int, optional
            Number of measurement shots.  If ``None`` the result is the
            exact expectation value from the statevector.
        noise : {'gaussian', 'poisson'}, optional
            Noise model applied when ``shots`` is provided.  Default is
            Gaussian.
        seed : int, optional
            Random seed for reproducible noise.
        """
        rng = np.random.default_rng(seed)
        results: List[List[complex]] = []

        for params in parameter_sets:
            circ = self._bind(params)
            state = Statevector.from_instruction(circ)
            row: List[complex] = [
                state.expectation_value(obs) for obs in observables
            ]
            results.append(row)

        if shots is not None:
            if noise is None or noise == "gaussian":
                std = max(1e-6, 1.0 / shots)
                results = [
                    [rng.normal(val.real, std) + 1j * rng.normal(val.imag, std)
                     for val in row]
                    for row in results
                ]
            elif noise == "poisson":
                results = [
                    [rng.poisson(val.real) / shots + 1j * rng.poisson(val.imag) / shots
                     for val in row]
                    for row in results
                ]
            else:
                raise ValueError(f"Unknown noise model: {noise}")

        return results
