"""Quantum estimator for parametrized circuits.

Enhancements:
- Gradient evaluation via parameter‑shift rule.
- Choice of backend (statevector or qasm).
- Optional shot‑noise simulation.
- Caching of expectation values for repeated parameter sets.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple, Dict, Optional

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli
from qiskit.quantum_info.operators import BaseOperator

class FastBaseEstimator:
    """Evaluates expectation values and gradients for a parametrized circuit."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: str | None = None,
        shots: int | None = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend_name = backend or ("statevector_simulator" if shots is None else "qasm_simulator")
        self.shots = shots
        self._cache: Dict[Tuple[float,...], List[complex]] = {}

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _expectation(self, circuit: QuantumCircuit, observable: BaseOperator) -> complex:
        if self.backend_name == "statevector_simulator":
            sv = Statevector.from_instruction(circuit)
            return sv.expectation_value(observable).real
        else:
            # Use qasm simulation with expectation estimation via measurement
            backend = Aer.get_backend(self.backend_name)
            job = execute(circuit, backend=backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts(circuit)
            # Only supports Pauli Z measurement for simplicity
            if not isinstance(observable, Pauli):
                raise TypeError("For qasm backend, observable must be a Pauli operator.")
            exp = 0.0
            for bitstring, count in counts.items():
                parity = (-1) ** (bitstring.count("1"))
                exp += parity * count
            exp /= self.shots
            return complex(exp)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots_noise: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        rng = np.random.default_rng(seed)
        for values in parameter_sets:
            key = tuple(values)
            if key in self._cache:
                row = self._cache[key]
            else:
                circuit = self._bind(values)
                row = [self._expectation(circuit, obs) for obs in observables]
                self._cache[key] = row
            if shots_noise is not None:
                noisy = [complex(rng.normal(val.real, max(1e-6, 1 / shots_noise))) for val in row]
                results.append(noisy)
            else:
                results.append(row)
        return results

    def gradient(
        self,
        observable: BaseOperator,
        parameter_sets: Sequence[Sequence[float]],
        *,
        stepsize: float = np.pi / 2,
    ) -> List[List[float]]:
        grads: List[List[float]] = []
        for values in parameter_sets:
            grad_row: List[float] = []
            for idx, _ in enumerate(values):
                shift_plus = list(values)
                shift_minus = list(values)
                shift_plus[idx] += stepsize
                shift_minus[idx] -= stepsize
                exp_plus = self._expectation(self._bind(shift_plus), observable)
                exp_minus = self._expectation(self._bind(shift_minus), observable)
                grad = (exp_plus - exp_minus) / (2 * np.sin(stepsize))
                grad_row.append(float(grad))
            grads.append(grad_row)
        return grads

    def evaluate_with_gradients(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        stepsize: float = np.pi / 2,
        shots_noise: int | None = None,
        seed: int | None = None,
    ) -> Tuple[List[List[complex]], List[List[List[float]]]]:
        values = self.evaluate(observables, parameter_sets, shots_noise=shots_noise, seed=seed)
        gradients: List[List[List[float]]] = []
        for obs in observables:
            gradients.append(self.gradient(obs, parameter_sets, stepsize=stepsize))
        # Transpose gradients to match row structure
        transposed = [
            [gradients[obs_idx][row_idx] for obs_idx in range(len(observables))]
            for row_idx in range(len(parameter_sets))
        ]
        return values, transposed


__all__ = ["FastBaseEstimator"]
