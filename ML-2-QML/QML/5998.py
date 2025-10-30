from __future__ import annotations

import numpy as np
from collections.abc import Iterable, Sequence
from typing import List, Optional

from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """
    Lightweight estimator for parameterised quantum circuits.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parametrised circuit to evaluate.
    backend : str | None, optional
        Backend name (default: ``'statevector_simulator'``).
    """

    def __init__(self, circuit: QuantumCircuit, backend: str | None = None) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend = backend or "statevector_simulator"
        self._simulator = Aer.get_backend(self.backend)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
            Operators for which expectation values are required.
        parameter_sets : Sequence[Sequence[float]]
            Sequence of parameter vectors.

        Returns
        -------
        List[List[complex]]
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            circuit = self._bind(values)
            state = Statevector.from_instruction(circuit)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_shots(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        shots: int,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Evaluate expectation values with shot noise using a backend that supports shots.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
        parameter_sets : Sequence[Sequence[float]]
        shots : int
            Number of shots for each evaluation.
        seed : Optional[int]
            Random seed for backend.

        Returns
        -------
        List[List[float]]
        """
        raw = self.evaluate(observables, parameter_sets)
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(val.real, max(1e-6, 1 / shots))) for val in row]
            noisy.append(noisy_row)
        return noisy

    def gradient(
        self,
        observable: BaseOperator,
        parameter_sets: Sequence[Sequence[float]],
        shift: float = np.pi / 2,
    ) -> List[List[float]]:
        """
        Compute the gradient of an observable w.r.t. circuit parameters using
        the parameter‑shift rule.

        Parameters
        ----------
        observable : BaseOperator
            Observable whose gradient is required.
        parameter_sets : Sequence[Sequence[float]]
            Parameter vectors.
        shift : float, default π/2
            Shift value for the parameter‑shift rule.

        Returns
        -------
        List[List[float]]
        """
        results: List[List[float]] = []
        for params in parameter_sets:
            grad_row: List[float] = []
            for i, _ in enumerate(params):
                plus = list(params)
                minus = list(params)
                plus[i] += shift
                minus[i] -= shift
                val_plus = self.evaluate([observable], [plus])[0][0].real
                val_minus = self.evaluate([observable], [minus])[0][0].real
                grad = 0.5 * (val_plus - val_minus)
                grad_row.append(grad)
            results.append(grad_row)
        return results


__all__ = ["FastBaseEstimator"]
