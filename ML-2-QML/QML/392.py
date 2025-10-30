"""Enhanced quantum estimator for parametrized circuits.

Features
--------
- Shot‑noise simulation via optional ``shots`` argument.
- Caching of bound circuits for repeated evaluations.
- Parameter‑shift gradient estimation for a single observable.
"""

from __future__ import annotations

import numpy as np
from collections.abc import Iterable, Sequence
from typing import List, Optional

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

class FastEstimatorV2:
    """Quantum estimator that evaluates expectation values with optional shot noise.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parametrised quantum circuit.
    device : str | None
        Target simulator backend. Currently only the built‑in
        Aer state‑vector simulator is used.
    """

    def __init__(self, circuit: QuantumCircuit, device: str | None = None) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)
        self.device = device or "aer_simulator"
        self._cache: dict[tuple[float,...], QuantumCircuit] = {}

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]],
                 *,
                 shots: int | None = None,
                 rng: np.random.Generator | None = None) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
            Operators to measure.
        parameter_sets : Sequence[Sequence[float]]
            Parameter vectors.
        shots : int | None
            Number of measurement shots. ``None`` means exact state‑vector
            expectation values.
        rng : np.random.Generator | None
            RNG used to simulate shot noise when ``shots`` is finite.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            key = tuple(params)
            if key in self._cache:
                bound = self._cache[key]
            else:
                bound = self._bind(params)
                self._cache[key] = bound

            if shots is None:
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                # Exact expectation values first
                state = Statevector.from_instruction(bound)
                ideal = [state.expectation_value(obs) for obs in observables]
                if rng is None:
                    rng = np.random.default_rng()
                # Gaussian shot noise with variance |exp|/shots
                noisy = [rng.normal(ideal_val, abs(ideal_val) / np.sqrt(shots)) for ideal_val in ideal]
                row = noisy
            results.append(row)
        return results

    def gradient(self,
                 observable: BaseOperator,
                 parameter_index: int,
                 parameter_set: Sequence[float],
                 *,
                 shift: float = np.pi / 2) -> float:
        """Estimate the gradient of an expectation value w.r.t a single parameter.

        Parameters
        ----------
        observable : BaseOperator
            Observable to measure.
        parameter_index : int
            Index of the parameter to differentiate.
        parameter_set : Sequence[float]
            Current parameter vector.
        shift : float
            Shift value for the parameter‑shift rule. Default is ``π/2``.
        """
        params_plus = list(parameter_set)
        params_minus = list(parameter_set)
        params_plus[parameter_index] += shift
        params_minus[parameter_index] -= shift

        val_plus = self.evaluate([observable], [params_plus], shots=None)[0][0]
        val_minus = self.evaluate([observable], [params_minus], shots=None)[0][0]
        return 0.5 * (val_plus - val_minus)

__all__ = ["FastEstimatorV2"]
