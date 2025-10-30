import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit import Aer
from collections.abc import Iterable, Sequence
from typing import List, Optional

BaseOperator = Operator


class FastBaseEstimator:
    """Variational circuit evaluator with optional gradient via parameter‑shift.

    Supports state‑vector simulation and backend execution with shot noise.
    """

    def __init__(self, circuit: QuantumCircuit, backend=None):
        self.circuit = circuit
        self.params = list(circuit.parameters)
        self.backend = backend or Aer.get_backend("statevector_simulator")

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self.params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.params, param_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Expectation values for each observable and parameter set."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = self._bind(values)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_with_gradients(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        shift: float = np.pi / 2,
    ) -> List[tuple[List[complex], List[np.ndarray]]]:
        """Return expectation values and gradients via parameter‑shift rule."""
        observables = list(observables)
        results: List[tuple[List[complex], List[np.ndarray]]] = []
        for values in parameter_sets:
            vals: List[complex] = []
            grads: List[np.ndarray] = []
            for obs in observables:
                exp = self._expectation(obs, values)
                vals.append(exp)
                grad = np.array(
                    [self._parameter_shift(obs, values, i, shift) for i in range(len(values))]
                )
                grads.append(grad)
            results.append((vals, grads))
        return results

    def _expectation(self, obs: BaseOperator, values: Sequence[float]) -> complex:
        bound = self._bind(values)
        state = Statevector.from_instruction(bound)
        return state.expectation_value(obs)

    def _parameter_shift(
        self,
        obs: BaseOperator,
        values: Sequence[float],
        idx: int,
        shift: float,
    ) -> complex:
        plus = list(values)
        minus = list(values)
        plus[idx] += shift
        minus[idx] -= shift
        exp_plus = self._expectation(obs, plus)
        exp_minus = self._expectation(obs, minus)
        return 0.5 * (exp_plus - exp_minus)

    def add_shot_noise(
        self,
        expectation_values: List[List[complex]],
        shots: int,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """Simulate shot noise by sampling from a Gaussian distribution."""
        rng = np.random.default_rng(seed)
        noisy = []
        for row in expectation_values:
            noisy_row = [
                rng.normal(exp.real, max(1e-6, 1 / shots))
                + 1j * rng.normal(exp.imag, max(1e-6, 1 / shots))
                for exp in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator"]
