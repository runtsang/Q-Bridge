"""Advanced hybrid estimator with shot noise, adaptive sampling, and support for multiple backends.

This module extends the lightweight quantum estimator by adding support for
simulating with a choice of backend (state‑vector or shot‑based) and by
providing an adaptive sampling routine that selects the most promising
parameter set from a candidate pool.  The API mirrors the classical
counterpart so that the two estimators can be used interchangeably.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer import AerSimulator
from qiskit import execute

# --------------------------------------------------------------------------- #
# Core estimator class
# --------------------------------------------------------------------------- #
class AdvancedHybridEstimator:
    """Evaluate expectation values of observables for a parametrized circuit,
    with optional shot noise and adaptive sampling.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parametrized quantum circuit whose parameters will be bound.
    backend : str, optional
        Backend name for simulation.  ``"statevector"`` (default) runs a
        state‑vector simulator; any Aer simulator name will run a
        shot‑based simulation.
    shots : int, optional
        Number of shots to use when ``backend`` is shot‑based.  If
        ``None`` and a shot‑based backend is requested, a default of
        1024 shots is used.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: str = "statevector",
        shots: Optional[int] = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._backend_name = backend
        self._shots = shots
        if backend == "statevector":
            self._simulator = AerSimulator(method="statevector")
        else:
            self._simulator = AerSimulator(method=backend)
        # Cache bound circuits for speed
        self._bound_circuits: dict[tuple[float,...], QuantumCircuit] = {}

    # --------------------------------------------------------------------- #
    # Private helpers
    # --------------------------------------------------------------------- #
    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        key = tuple(parameter_values)
        if key not in self._bound_circuits:
            self._bound_circuits[key] = self._circuit.assign_parameters(mapping, inplace=False)
        return self._bound_circuits[key]

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : iterable of BaseOperator
            Operators whose expectation values are to be evaluated.
        parameter_sets : sequence of sequences
            List of parameter vectors to evaluate.

        Returns
        -------
        List[List[complex]]
            A matrix of shape ``(n_params, n_observables)``.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound = self._bind(values)
            if self._backend_name == "statevector":
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                job = execute(bound, self._simulator, shots=self._shots or 1024)
                result = job.result()
                counts = result.get_counts(bound)
                state = Statevector.from_counts(counts)
                row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    # --------------------------------------------------------------------- #
    # Adaptive sampling
    # --------------------------------------------------------------------- #
    def select_next(
        self,
        prior_mean: Sequence[float],
        prior_cov: Sequence[Sequence[float]],
        candidate_sets: Sequence[Sequence[float]],
        observables: Sequence[BaseOperator],
        metric: Callable[[complex], float] = lambda x: x.real,
    ) -> Sequence[float]:
        """
        Choose the next parameter set to evaluate by maximizing an
        acquisition metric (default: real part of the first observable).

        Parameters
        ----------
        prior_mean : sequence of floats
            Current posterior mean over the parameter space.
        prior_cov : sequence of sequences
            Current posterior covariance over the parameter space.
        candidate_sets : sequence of sequences
            List of parameter vectors to consider.
        observables : sequence of BaseOperator
            Operators used to compute the acquisition metric.
        metric : callable, optional
            Function that maps a scalar expectation value to a
            acquisition value.

        Returns
        -------
        Sequence[float]
            The parameter vector with the highest acquisition value.
        """
        best_val = -np.inf
        best_params: Optional[Sequence[float]] = None

        for params in candidate_sets:
            bound = self._bind(params)
            if self._backend_name == "statevector":
                state = Statevector.from_instruction(bound)
                score = metric(state.expectation_value(observables[0]))
            else:
                job = execute(bound, self._simulator, shots=self._shots or 1024)
                result = job.result()
                counts = result.get_counts(bound)
                state = Statevector.from_counts(counts)
                score = metric(state.expectation_value(observables[0]))
            if score > best_val:
                best_val = score
                best_params = list(params)

        if best_params is None:
            raise RuntimeError("No candidate parameters were supplied.")
        return best_params

__all__ = ["AdvancedHybridEstimator"]
