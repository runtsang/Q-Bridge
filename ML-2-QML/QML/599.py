"""Quantum estimator with shot‑noise, parameter‑shift gradients, and backend flexibility."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Dict, Iterable, List, Tuple

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.parameter import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

class FastBaseEstimator:
    """Quantum expectation value estimator with shot‑noise and gradient support.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parametrised quantum circuit.
    shots : int | None, default=None
        Number of shots; if None, use state‑vector simulator for exact results.
    backend : str | None, default=None
        Name of Aer simulator backend; if None and shots is not None, use Aer.get_backend('qasm_simulator').
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        shots: Optional[int] = None,
        backend: Optional[str] = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.shots = shots
        self.backend = backend or ("qasm_simulator" if shots is not None else "statevector_simulator")
        self._sim = Aer.get_backend(self.backend)

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
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound_qc = self._bind(values)
            if self.shots is None:
                state = Statevector.from_instruction(bound_qc)
                row = [state.expectation_value(obs).real for obs in observables]
            else:
                job = execute(bound_qc, self._sim, shots=self.shots, memory=True)
                counts = job.result().get_counts()
                # Convert counts to statevector probabilities
                probs = {bitstring: count / self.shots for bitstring, count in counts.items()}
                row = [self._expectation_from_counts(obs, probs) for obs in observables]
            results.append(row)
        return results

    def _expectation_from_counts(self, observable: BaseOperator, probs: Dict[str, float]) -> complex:
        """Compute expectation value from measurement counts."""
        exp = 0.0
        for bitstring, prob in probs.items():
            exp += prob * observable.expectation_value(Statevector(bitstring))
        return exp

    def gradient(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[Tuple[float, float]]]:
        """Estimate gradients using the parameter‑shift rule.

        Returns
        -------
        List[List[Tuple[float, float]]]
            Gradients for each observable and parameter: (shifted_plus, shifted_minus).
        """
        observables = list(observables)
        grad_results: List[List[Tuple[float, float]]] = []
        shift = np.pi / 2
        for values in parameter_sets:
            grads_per_obs: List[Tuple[float, float]] = []
            for obs in observables:
                grad_sum = 0.0
                for idx, param in enumerate(values):
                    plus = list(values)
                    minus = list(values)
                    plus[idx] += shift
                    minus[idx] -= shift
                    exp_plus = self._expectation_from_counts(obs, self._measure_counts(plus))
                    exp_minus = self._expectation_from_counts(obs, self._measure_counts(minus))
                    grad_sum += (exp_plus - exp_minus) / 2
                grads_per_obs.append((grad_sum, 0.0))  # second element reserved for variance
            grad_results.append(grads_per_obs)
        return grad_results

    def _measure_counts(self, parameter_values: Sequence[float]) -> Dict[str, float]:
        bound_qc = self._bind(parameter_values)
        job = execute(bound_qc, self._sim, shots=self.shots or 1000, memory=True)
        counts = job.result().get_counts()
        return {bitstring: count / (self.shots or 1000) for bitstring, count in counts.items()}

__all__ = ["FastBaseEstimator"]
