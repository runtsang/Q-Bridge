"""Enhanced estimator primitive for Qiskit circuits with shot noise and gradients."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable, List, Optional, Dict, Any

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer import AerSimulator
from qiskit.providers import BackendV1


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit with optional backend and shots."""

    def __init__(self, circuit: QuantumCircuit, backend: Optional[BackendV1] = None) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend = backend or AerSimulator(method="statevector")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        backend: Optional[BackendV1] = None,
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        backend = backend or self.backend

        for values in parameter_sets:
            bound_circ = self._bind(values)
            if shots is None:
                state = Statevector.from_instruction(bound_circ)
                row = [state.expectation_value(observable) for observable in observables]
            else:
                simulator = backend if isinstance(backend, BackendV1) else AerSimulator()
                job = simulator.run(bound_circ, shots=shots)
                result = job.result()
                counts = result.get_counts()
                probs = {k: v / shots for k, v in counts.items()}
                # Use the most probable outcome as a crude statevector approximation
                state = Statevector.from_label(max(probs, key=probs.get))
                row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results

    def evaluate_shots(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int,
        backend: Optional[BackendV1] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """Simulate measurement shot noise by sampling from the probability distribution."""
        rng = np.random.default_rng(seed)
        backend = backend or self.backend
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound_circ = self._bind(values)
            simulator = backend if isinstance(backend, BackendV1) else AerSimulator()
            job = simulator.run(bound_circ, shots=shots)
            result = job.result()
            counts = result.get_counts()
            probs = {k: v / shots for k, v in counts.items()}
            sampled_labels = rng.choice(list(probs.keys()), size=shots, p=list(probs.values()))
            hist: Dict[str, int] = {}
            for lbl in sampled_labels:
                hist[lbl] = hist.get(lbl, 0) + 1
            row: List[complex] = []
            for observable in observables:
                exp_val = 0.0
                for bitstring, count in hist.items():
                    sv = Statevector.from_label(bitstring)
                    exp_val += count * sv.expectation_value(observable).real
                exp_val /= shots
                row.append(complex(exp_val))
            results.append(row)
        return results

    def evaluate_grad(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shift: float = np.pi / 2,
    ) -> List[List[List[complex]]]:
        """Compute gradients of expectation values using the parameter‑shift rule."""
        observables = list(observables)
        grads: List[List[List[complex]]] = []

        for values in parameter_sets:
            grad_row: List[List[complex]] = []
            for obs in observables:
                grad_vals: List[complex] = []
                for idx, _ in enumerate(values):
                    shift_plus = list(values)
                    shift_minus = list(values)
                    shift_plus[idx] += shift
                    shift_minus[idx] -= shift
                    exp_plus = self.evaluate([obs], [shift_plus], shots=None)[0][0]
                    exp_minus = self.evaluate([obs], [shift_minus], shots=None)[0][0]
                    grad = (exp_plus - exp_minus) / (2 * np.sin(shift))
                    grad_vals.append(grad)
                grad_row.append(grad_vals)
            grads.append(grad_row)
        return grads


__all__ = ["FastBaseEstimator"]
