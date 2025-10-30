"""Minimal estimator primitive used by the simplified fast primitives package, extended with shot‑noise simulation, batched evaluation, and parameter‑shift gradient support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional, Tuple

import numpy as np
from qiskit import Aer, execute, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.providers import Backend


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit, with batched evaluation and optional noise."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._backend = Aer.get_backend("statevector_simulator")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        noise_model: Optional[object] = None,
        backend: Optional[Backend] = None,
        optimization_level: int = 0,
    ) -> List[List[complex]]:
        """
        Compute expectation values for every parameter set and observable.
        Supports optional shot noise and noise models.

        Parameters
        ----------
        observables
            Iterable of qiskit Operator instances.
        parameter_sets
            Sequence of parameter vectors.
        shots
            Number of measurement shots. If None, use statevector simulation.
        noise_model
            Optional qiskit noise model.
        backend
            Backend to execute on. Defaults to a statevector simulator.
        optimization_level
            Optimization level for transpilation.

        Returns
        -------
        List[List[complex]]
            Nested list of expectation values.
        """
        observables = list(observables)
        if backend is None:
            backend = self._backend
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound = self._bind(values)
            if shots is None:
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                transpiled = transpile(bound, backend=backend, optimization_level=optimization_level)
                job = execute(
                    transpiled,
                    backend=backend,
                    shots=shots,
                    noise_model=noise_model,
                )
                result = job.result()
                counts = result.get_counts()
                # Build density matrix from counts
                statevec = Statevector.from_counts(counts)
                density = np.outer(statevec.data, statevec.data.conj())
                row = [np.trace(density @ obs.data).real for obs in observables]
            results.append(row)
        return results

    def evaluate_with_grad(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        backend: Optional[Backend] = None,
        optimization_level: int = 0,
    ) -> List[List[np.ndarray]]:
        """
        Compute gradients of expectation values with respect to parameters using the parameter‑shift rule.
        Returns a list of gradient arrays of shape (num_params, num_observables).

        Parameters
        ----------
        observables
            Iterable of qiskit Operator instances.
        parameter_sets
            Sequence of parameter vectors.
        backend
            Backend to execute on. Defaults to a statevector simulator.
        optimization_level
            Optimization level for transpilation.

        Returns
        -------
        List[List[np.ndarray]]
            Each element corresponds to a parameter set and contains an array of shape
            (num_params, num_observables) with the gradients.
        """
        observables = list(observables)
        if backend is None:
            backend = self._backend
        grads: List[List[np.ndarray]] = []
        shift = np.pi / 2

        for values in parameter_sets:
            grad_per_obs: List[np.ndarray] = []
            for obs in observables:
                grad_vec = []
                for idx, val in enumerate(values):
                    plus = list(values)
                    minus = list(values)
                    plus[idx] += shift
                    minus[idx] -= shift
                    exp_plus = self.evaluate(
                        [obs],
                        [plus],
                        shots=None,
                        backend=backend,
                        optimization_level=optimization_level,
                    )[0][0]
                    exp_minus = self.evaluate(
                        [obs],
                        [minus],
                        shots=None,
                        backend=backend,
                        optimization_level=optimization_level,
                    )[0][0]
                    grad = (exp_plus - exp_minus) / 2
                    grad_vec.append(grad)
                grad_per_obs.append(np.array(grad_vec))
            grads.append(grad_per_obs)
        return grads


__all__ = ["FastBaseEstimator"]
