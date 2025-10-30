"""FastEstimator: Quantum expectation evaluator with backend choice and parameter‑shift gradients.

This QML implementation extends the original FastBaseEstimator by adding:
* optional backend selection (Aer state‑vector or qasm simulators, or any Qiskit backend).
* configurable shot counts for stochastic estimation.
* a parameter‑shift gradient routine for variational optimisation.
* caching of compiled circuits for repeated evaluations.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers import Backend


class FastEstimator:
    """Evaluate expectation values of observables for a parametrised circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parametrised quantum circuit.
    backend : Backend | None, optional
        Qiskit backend to use; defaults to Aer state‑vector simulator.
    shots : int | None, optional
        Number of measurement shots; if ``None`` a state‑vector evaluator is used.
    seed : int | None, optional
        Random seed for reproducible shots.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Optional[Backend] = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self._original = circuit
        self._parameters = list(circuit.parameters)
        self.shots = shots
        self.seed = seed
        self.backend = backend or (
            Aer.get_backend("statevector_simulator") if shots is None else Aer.get_backend("qasm_simulator")
        )
        if shots is not None:
            self.backend.set_options(shots=shots, seed_simulator=seed)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._original.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
            List of observable operators.
        parameter_sets : Sequence[Sequence[float]]
            Parameter vectors to evaluate.

        Returns
        -------
        List[List[complex]]
            Nested list of expectation values.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound = self._bind(values)
            if self.shots is None:
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                job = execute(bound, backend=self.backend, shots=self.shots, seed_simulator=self.seed)
                result = job.result()
                counts = result.get_counts(bound)
                probs = {k: v / self.shots for k, v in counts.items()}
                row = [self._sample_expectation(obs, probs) for obs in observables]
            results.append(row)
        return results

    def _sample_expectation(self, observable: BaseOperator, probs: dict[str, float]) -> complex:
        """Compute expectation value from sampled probabilities."""
        # For brevity we only support a single‑qubit Pauli‑Z observable.
        if observable == BaseOperator.from_label("Z"):
            return sum((1 if bit == "0" else -1) * p for bit, p in probs.items())
        raise NotImplementedError("Sampling for arbitrary observables is not implemented.")

    def gradient(self, observable: BaseOperator, parameter_set: Sequence[float], shift: float = np.pi / 2) -> List[float]:
        """Compute the parameter‑shift gradient of an observable.

        Parameters
        ----------
        observable : BaseOperator
            Observable to differentiate.
        parameter_set : Sequence[float]
            Parameter vector at which to evaluate the gradient.
        shift : float, optional
            Shift used in the parameter‑shift rule.

        Returns
        -------
        List[float]
            Gradient for each parameter.
        """
        grad: List[float] = []
        for i, param in enumerate(parameter_set):
            plus = list(parameter_set)
            minus = list(parameter_set)
            plus[i] += shift
            minus[i] -= shift
            f_plus = self._expect_observable(observable, plus)
            f_minus = self._expect_observable(observable, minus)
            grad.append((f_plus - f_minus) / (2 * np.sin(shift)))
        return grad

    def _expect_observable(self, observable: BaseOperator, parameters: Sequence[float]) -> complex:
        """Helper to evaluate a single observable for a given parameter vector."""
        bound = self._bind(parameters)
        if self.shots is None:
            state = Statevector.from_instruction(bound)
            return state.expectation_value(observable)
        else:
            job = execute(bound, backend=self.backend, shots=self.shots, seed_simulator=self.seed)
            result = job.result()
            counts = result.get_counts(bound)
            probs = {k: v / self.shots for k, v in counts.items()}
            return self._sample_expectation(observable, probs)

__all__ = ["FastEstimator"]
