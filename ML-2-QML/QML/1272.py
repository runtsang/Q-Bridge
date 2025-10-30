"""Minimal estimator primitive used by the simplified fast primitives package,
extended with analytic gradient support and shot‑noise emulation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""

    def __init__(self, circuit: QuantumCircuit, backend_name: str = "statevector_simulator") -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._backend = Aer.get_backend(backend_name)

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
    ) -> np.ndarray:
        """Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
            A list‑like of operators to evaluate.
        parameter_sets : Sequence[Sequence[float]]
            A sequence of parameter vectors.
        shots : int, optional
            If provided, the results are perturbed with Gaussian noise
            of variance 1/shots to emulate shot noise.
        seed : int, optional
            Random seed for shot‑noise simulation.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound_circuit = self._bind(values)
            if shots is None:
                state = Statevector.from_instruction(bound_circuit)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                job = execute(
                    bound_circuit,
                    self._backend,
                    shots=shots,
                    seed_simulator=seed,
                )
                result = job.result()
                state = Statevector.from_result(result)
                row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        array = np.array(results, dtype=np.complex128)
        if shots is not None:
            rng = np.random.default_rng(seed)
            noise = rng.normal(0, np.sqrt(1 / shots), size=array.shape)
            array = array + noise
        return array

    def gradient(
        self,
        observable: BaseOperator,
        parameter_sets: Sequence[Sequence[float]],
        shift: float = np.pi / 2,
    ) -> np.ndarray:
        """Compute analytic gradients using the parameter‑shift rule.

        Parameters
        ----------
        observable : BaseOperator
            The operator whose expectation value gradient is desired.
        parameter_sets : Sequence[Sequence[float]]
            A sequence of parameter vectors.
        shift : float, default π/2
            The shift angle used in the parameter‑shift rule.
        """
        grads: List[List[float]] = []
        for values in parameter_sets:
            plus = list(values)
            minus = list(values)
            for i, _ in enumerate(values):
                plus[i] += shift
                minus[i] -= shift
            plus_circ = self._bind(plus)
            minus_circ = self._bind(minus)
            f_plus = Statevector.from_instruction(plus_circ).expectation_value(observable)
            f_minus = Statevector.from_instruction(minus_circ).expectation_value(observable)
            grad = (f_plus - f_minus) / (2 * np.sin(shift))
            grads.append(grad.real.tolist())
        return np.array(grads, dtype=np.float32)


__all__ = ["FastBaseEstimator"]
