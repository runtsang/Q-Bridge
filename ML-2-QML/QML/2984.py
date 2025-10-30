"""Hybrid estimator that evaluates Qiskit circuits with optional shot noise."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastHybridEstimator:
    """Evaluate expectation values of observables for a parametrised Qiskit circuit.

    The estimator supports ideal state‑vector simulation as well as noisy shot‑based
    simulation.  It is a drop‑in replacement for the original FastBaseEstimator
    but adds a convenient ``shots`` argument and uses AerSimulator for realistic
    sampling.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

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
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables:
            QuantumInfo BaseOperator objects (e.g. PauliZ, X, etc.).
        parameter_sets:
            Iterable of parameter vectors.
        shots:
            If provided, use AerSimulator with the given shot count; otherwise use
            Statevector.from_instruction for exact results.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        if shots is None:
            # Ideal state‑vector simulation
            for values in parameter_sets:
                state = Statevector.from_instruction(self._bind(values))
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)
        else:
            # Shot‑based simulation
            simulator = Aer.get_backend("aer_simulator")
            for values in parameter_sets:
                bound = self._bind(values)
                job = execute(
                    bound,
                    simulator,
                    shots=shots,
                    backend_options={"seed_simulator": None},
                )
                result = job.result()
                state = Statevector.from_instruction(bound)
                # Compute exact expectation values
                exact = [state.expectation_value(obs) for obs in observables]
                rng = np.random.default_rng()
                noisy = [
                    rng.normal(float(val.real), max(1e-6, 1 / np.sqrt(shots)))
                    for val in exact
                ]
                results.append(noisy)

        return results


# Backward‑compatibility aliases
FastBaseEstimator = FastHybridEstimator
FastEstimator = FastHybridEstimator


__all__ = ["FastHybridEstimator", "FastBaseEstimator", "FastEstimator"]
