from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
from qiskit import Aer
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """Fast evaluation of parametrised quantum circuits.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parametrised circuit whose parameters will be bound during evaluation.
    backend : str, optional
        Backend name. ``'statevector_simulator'`` (default) or ``'qasm_simulator'``.
    shots : int | None, optional
        Number of shots for shot‑based estimation. Ignored for the state‑vector backend.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: str = "statevector_simulator",
        shots: int | None = None,
    ) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)
        self.backend_name = backend
        self.shots = shots
        if shots is not None and backend == "qasm_simulator":
            self.backend = Aer.get_backend("qasm_simulator")
        else:
            self.backend = None

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def _expect_statevector(self, state: Statevector, observable: BaseOperator) -> complex:
        return state.expectation_value(observable)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each observable and parameter set."""
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [self._expect_statevector(state, obs) for obs in observables]
            if self.shots is not None:
                rng = np.random.default_rng()
                noisy_row = [
                    val + rng.normal(0, 1 / np.sqrt(self.shots))
                    for val in row
                ]
                row = noisy_row
            results.append(row)

        return results

    def evaluate_with_gradients(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> Tuple[List[List[complex]], List[List[np.ndarray]]]:
        """Compute expectation values and gradients w.r.t circuit parameters
        using the parameter‑shift rule."""
        observables = list(observables)
        expectations: List[List[complex]] = []
        gradients: List[List[np.ndarray]] = []
        shift = np.pi / 2

        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            exp_row: List[complex] = []
            grad_row: List[np.ndarray] = []

            for obs in observables:
                exp_val = state.expectation_value(obs)
                exp_row.append(exp_val)

                grad_vec: List[float] = []
                for i in range(len(values)):
                    forward = list(values)
                    forward[i] += shift
                    backward = list(values)
                    backward[i] -= shift
                    f_state = Statevector.from_instruction(self._bind(forward))
                    b_state = Statevector.from_instruction(self._bind(backward))
                    f_val = f_state.expectation_value(obs)
                    b_val = b_state.expectation_value(obs)
                    grad = 0.5 * (f_val - b_val)
                    grad_vec.append(float(grad.real))
                grad_row.append(np.array(grad_vec, dtype=np.float64))
            expectations.append(exp_row)
            gradients.append(grad_row)

        return expectations, gradients


__all__ = ["FastBaseEstimator"]
