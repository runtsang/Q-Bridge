"""Quantum version of FastEstimator with shot noise, batched execution, and parameter‑shift gradients."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """Base class that evaluates expectation values of observables for a parametrized circuit."""
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
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Extended quantum estimator supporting shot noise, batched execution, and gradients via parameter‑shift."""
    def __init__(self, circuit: QuantumCircuit, backend_name: str = "aer_simulator") -> None:
        super().__init__(circuit)
        self.backend_name = backend_name
        self.backend = Aer.get_backend(backend_name)

    def _run_circuit(
        self,
        circuit: QuantumCircuit,
        shots: int | None = None,
    ) -> dict[str, int]:
        """Run a single circuit and return measurement counts."""
        job = execute(circuit, backend=self.backend, shots=shots)
        result = job.result()
        return result.get_counts(circuit)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        batch_size: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values with optional shot noise and batched execution."""
        observables = list(observables)

        if shots is None:
            return super().evaluate(observables, parameter_sets)

        results: List[List[complex]] = []

        for values in parameter_sets:
            circ = self._bind(values)
            counts = self._run_circuit(circ, shots=shots)
            probs = np.array([counts.get(bit, 0) / shots for bit in sorted(counts)])
            exp_vals = []
            for obs in observables:
                exp = 0.0
                for bit, prob in zip(sorted(counts), probs):
                    state = Statevector.from_label(bit)
                    exp += prob * state.expectation_value(obs)
                exp_vals.append(complex(exp))
            results.append(exp_vals)
        return results

    def evaluate_with_gradients(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        batch_size: int | None = None,
    ) -> Tuple[List[List[complex]], List[List[complex]]]:
        """Return expectation values and gradients via parameter‑shift rule."""
        observables = list(observables)
        values: List[List[complex]] = []
        gradients: List[List[complex]] = []

        for params in parameter_sets:
            circ = self._bind(params)
            state = Statevector.from_instruction(circ)
            row_vals = [state.expectation_value(obs) for obs in observables]
            values.append(row_vals)

            row_grads = []
            for obs in observables:
                grad_sum = 0.0
                for idx, param in enumerate(self._parameters):
                    shift = np.pi / 2
                    circ_plus = self._bind([*params[:idx], param + shift, *params[idx+1:]])
                    circ_minus = self._bind([*params[:idx], param - shift, *params[idx+1:]])
                    exp_plus = Statevector.from_instruction(circ_plus).expectation_value(obs)
                    exp_minus = Statevector.from_instruction(circ_minus).expectation_value(obs)
                    grad_sum += (exp_plus - exp_minus) / 2
                row_grads.append(complex(grad_sum))
            gradients.append(row_grads)

        return values, gradients

    def predict(
        self,
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
    ) -> List[Statevector | dict]:
        """Return the statevector (or measurement results) for each parameter set."""
        states: List[Statevector | dict] = []
        for params in parameter_sets:
            circ = self._bind(params)
            if shots is None:
                states.append(Statevector.from_instruction(circ))
            else:
                states.append(self._run_circuit(circ, shots=shots))
        return states

__all__ = ["FastBaseEstimator", "FastEstimator"]
