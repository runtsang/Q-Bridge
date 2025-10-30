from __future__ import annotations

import numpy as np
from typing import Iterable, List, Sequence
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

class FastBaseEstimator:
    """
    Lightweight estimator that evaluates expectation values of quantum
    observables for a parameterised circuit.  Supports exact statevector
    evaluation and optional shot-noise emulation.
    """
    def __init__(self, circuit: QuantumCircuit, backend: str = "statevector") -> None:
        """
        Parameters
        ----------
        circuit:
            Parameterised `QuantumCircuit` whose parameters will be bound per call.
        backend:
            The evaluation backend; currently only'statevector' is supported.
        """
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._backend = backend

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Parameters
        ----------
        observables:
            Iterable of `BaseOperator` objects whose expectation values are
            computed.
        parameter_sets:
            Iterable of parameter sequences to bind to the circuit.
        shots:
            If provided, Gaussian noise with std = 1 / sqrt(shots) is added to
            each expectation value to emulate finite sampling.
        seed:
            Random seed for reproducibility of the shot-noise emulation.
        Returns
        -------
        List of lists of complex expectation values, one list per parameter set.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            circuit = self._bind(values)
            state = Statevector.from_instruction(circuit)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [
                complex(rng.normal(np.real(val), max(1e-6, 1 / shots)),
                        rng.normal(np.imag(val), max(1e-6, 1 / shots)))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy

__all__ = ["FastBaseEstimator"]
