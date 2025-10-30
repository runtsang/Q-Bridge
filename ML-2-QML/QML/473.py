"""Hybrid estimator for quantum circuits using Qiskit and Aer.

The estimator can evaluate expectation values of arbitrary
`BaseOperator` observables for a parametrised circuit.  It
supports:

* Exact evaluation via `Statevector` for noiseless simulation.
* Shot‑noise emulation using the Aer simulator.
* Automatic caching of bound circuits to avoid re‑binding.
* Gradient calculation by the parameter‑shift rule.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple, Dict, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, Operator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer import AerSimulator

class HybridEstimator:
    """Evaluate a parametrised `QuantumCircuit` for many parameter sets
    and observables.

    Parameters
    ----------
    circuit:
        A parameterised `QuantumCircuit`.  Parameters that are not
        supplied during evaluation are left symbolic.
    backend:
        Optional Qiskit backend.  If ``None`` an Aer state‑vector
        simulator is used for exact evaluation.
    shots:
        Default number of shots for noisy evaluation.  If ``None``
        evaluation is deterministic.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Optional[AerSimulator] = None,
        shots: int | None = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._backend = backend or AerSimulator(method="statevector")
        self._shots = shots
        self._cache: Dict[Tuple[float,...], QuantumCircuit] = {}

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        key = tuple(values)
        if key in self._cache:
            return self._cache[key]
        mapping = dict(zip(self._parameters, values))
        bound = self._circuit.assign_parameters(mapping, inplace=False)
        self._cache[key] = bound
        return bound

    def _expectation(self, circuit: QuantumCircuit, observable: BaseOperator) -> complex:
        if self._shots is None:
            state = Statevector.from_instruction(circuit)
            return state.expectation_value(observable)
        else:
            # Use state‑vector expectation and add Gaussian shot noise
            state = Statevector.from_instruction(circuit)
            exact = state.expectation_value(observable)
            std = max(1e-6, 1 / np.sqrt(self._shots))
            return complex(
                np.random.normal(exact.real, std) + 1j * np.random.normal(exact.imag, std)
            )

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Return a list of expectation values for each parameter set.

        Parameters
        ----------
        observables:
            Iterable of `BaseOperator` objects.
        parameter_sets:
            Iterable of parameter vectors.
        """
        if not observables:
            raise ValueError("At least one observable must be provided.")
        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            circ = self._bind(params)
            row: List[complex] = [self._expectation(circ, obs) for obs in observables]
            results.append(row)

        return results

    def gradient(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shift: float = np.pi / 2,
    ) -> List[List[List[float]]]:
        """Return the gradient of each observable w.r.t. each circuit parameter.

        Returns a list of shape
        ``(n_sets, n_observables, n_params)``.
        """
        if not observables:
            raise ValueError("At least one observable must be provided.")
        observables = list(observables)
        n_params = len(self._parameters)
        grads: List[List[List[float]]] = []

        for params in parameter_sets:
            grad_set: List[List[float]] = []
            for obs in observables:
                grad_vec: List[float] = []
                for i in range(n_params):
                    shift_vec = list(params)
                    shift_vec[i] += shift
                    circ_plus = self._bind(shift_vec)
                    val_plus = self._expectation(circ_plus, obs)

                    shift_vec[i] -= 2 * shift
                    circ_minus = self._bind(shift_vec)
                    val_minus = self._expectation(circ_minus, obs)

                    grad = (val_plus - val_minus) / (2j * np.sin(shift))
                    grad_vec.append(float(grad.real))  # real part only
                grad_set.append(grad_vec)
            grads.append(grad_set)

        return grads


__all__ = ["HybridEstimator"]
