"""Quantum estimator with simulator selection and gradient evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple, Optional, Callable
import numpy as np

from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.opflow import PauliSumOp

ScalarObservable = Callable[[Statevector], complex]
GradientObservable = Callable[[Statevector], np.ndarray]

class FastBaseEstimator:
    """Estimator for parameterized quantum circuits.

    Supports multiple simulators, shot noise, and analytic gradients via
    the parameter-shift rule.
    """
    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: str | Aer = "statevector",
        shots: Optional[int] = None,
    ) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)
        self.shots = shots
        if isinstance(backend, str):
            if backend == "statevector":
                self.backend = Aer.get_backend("statevector_simulator")
            elif backend == "qasm":
                self.backend = Aer.get_backend("qasm_simulator")
            else:
                raise ValueError(f"Unsupported backend {backend}")
        else:
            self.backend = backend

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, param_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Operator | PauliSumOp],
        parameter_sets: Sequence[Sequence[float]],
        *,
        compute_gradients: bool = False,
        gradient_observables: Optional[Iterable[Operator | PauliSumOp]] = None,
    ) -> Tuple[List[List[complex]], Optional[List[List[np.ndarray]]]]:
        observables = list(observables)
        if compute_gradients and gradient_observables is None:
            gradient_observables = observables

        results: List[List[complex]] = []
        gradients: List[List[np.ndarray]] | None = None

        for params in parameter_sets:
            circ = self._bind(params)
            if isinstance(self.backend, Aer):
                if self.backend.name == "statevector_simulator":
                    state = Statevector.from_instruction(circ)
                    row = [state.expectation_value(obs) for obs in observables]
                else:  # qasm simulator
                    job = execute(circ, self.backend, shots=self.shots or 1024)
                    result = job.result()
                    counts = result.get_counts()
                    row = [self._classical_expectation(counts, obs) for obs in observables]
            else:
                state = Statevector.from_instruction(circ)
                row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if compute_gradients:
            gradients = []
            for params in parameter_sets:
                grad_row: List[np.ndarray] = []
                for grad_obs in gradient_observables:
                    grad = self._parameter_shift_gradient(params, grad_obs)
                    grad_row.append(grad)
                gradients.append(grad_row)

        return results, gradients

    def _classical_expectation(self, counts: dict[str, int], observable: Operator) -> complex:
        total = sum(counts.values())
        exp = 0.0
        for bitstring, freq in counts.items():
            parity = (-1) ** sum(int(b) for b in bitstring)
            exp += parity * freq / total
        return exp

    def _parameter_shift_gradient(
        self,
        param_values: Sequence[float],
        observable: Operator,
    ) -> np.ndarray:
        grad = np.zeros(len(self.parameters))
        shift = np.pi / 2
        for i, _ in enumerate(self.parameters):
            circ_plus = self._bind([v + shift if j == i else v for j, v in enumerate(param_values)])
            circ_minus = self._bind([v - shift if j == i else v for j, v in enumerate(param_values)])

            state_plus = Statevector.from_instruction(circ_plus)
            state_minus = Statevector.from_instruction(circ_minus)

            exp_plus = state_plus.expectation_value(observable)
            exp_minus = state_minus.expectation_value(observable)

            grad[i] = 0.5 * (exp_plus - exp_minus).real
        return grad

__all__ = ["FastBaseEstimator"]
