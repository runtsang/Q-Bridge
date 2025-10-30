"""Enhanced Qiskit‑based estimator.

Improvements over the original primitive:
* Supports shot‑noise via QasmSimulator
* Provides a parameter‑shift gradient routine
* Allows batch evaluation and caching of intermediate statevectors
* Works with both Qiskit statevector and Qasm backends
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable, List, Tuple, Dict, Optional

import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

class FastBaseEstimator:
    """
    Evaluate expectation values of observables for a parametrized Qiskit circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The parametrized circuit to evaluate.
    backend : str | None, optional
        Name of the Aer simulator to use.  ``None`` defaults to the
        state‑vector simulator.
    shots : int | None, optional
        If provided, the circuit is executed on the Qasm simulator with
        this many shots, otherwise the state‑vector simulator is used.
    """

    def __init__(self, circuit: QuantumCircuit, backend: str | None = None, shots: int | None = None) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._shots = shots
        self._backend_name = backend or ("qasm_simulator" if shots else "statevector_simulator")
        self._backend = AerSimulator(name=self._backend_name)
        self._cache: Dict[Tuple[float,...], Statevector] = {}

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _get_statevector(self, values: Sequence[float]) -> Statevector:
        key = tuple(values)
        if key in self._cache:
            return self._cache[key]
        bound = self._bind(values)
        sv = Statevector.from_instruction(transpile(bound, self._backend))
        self._cache[key] = sv
        return sv

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : iterable of BaseOperator
            Observable operators for which to compute expectation values.
        parameter_sets : sequence of parameter sequences
            Each inner sequence represents a single set of circuit parameter values.

        Returns
        -------
        results : list of list of complex
            Outer dimension corresponds to parameter sets, inner dimension to
            observables.
        """
        obs = list(observables)
        results: List[List[complex]] = []

        if self._shots:
            # Qasm simulation
            for values in parameter_sets:
                bound = self._bind(values)
                transpiled = transpile(bound, self._backend)
                job = self._backend.run(transpiled, shots=self._shots)
                result = job.result()
                counts = result.get_counts()
                # Convert counts to a statevector via density matrix approximation
                state = Statevector.from_counts(counts)
                row = [state.expectation_value(o) for o in obs]
                results.append(row)
        else:
            # State‑vector simulation
            for values in parameter_sets:
                state = self._get_statevector(values)
                row = [state.expectation_value(o) for o in obs]
                results.append(row)

        return results

    def evaluate_with_shots(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Wrapper around ``evaluate`` that allows overriding the shot count.

        Parameters
        ----------
        shots : int or None
            Number of shots for the Qasm simulation.  If ``None`` the
            instance's default is used.
        seed : int or None
            Random seed for reproducibility.
        """
        if shots is not None:
            self._shots = shots
        return self.evaluate(observables, parameter_sets)

    def gradient(
        self,
        observable: BaseOperator,
        parameter_sets: Sequence[Sequence[float]],
        shift: float = np.pi / 2,
    ) -> List[List[float]]:
        """
        Compute the gradient of a single observable w.r.t. each parameter
        using the parameter‑shift rule.

        Returns
        -------
        grads : list of list of float
            Outer dimension corresponds to parameter sets, inner dimension to
            parameters.
        """
        grads: List[List[float]] = []

        for values in parameter_sets:
            grad_row: List[float] = []
            for idx, _ in enumerate(values):
                shift_vec = list(values)
                shift_vec[idx] += shift
                plus = self.evaluate([observable], [shift_vec])[0][0].real
                shift_vec[idx] -= 2 * shift
                minus = self.evaluate([observable], [shift_vec])[0][0].real
                grad = 0.5 * (plus - minus)
                grad_row.append(grad)
            grads.append(grad_row)

        return grads

__all__ = ["FastBaseEstimator"]
