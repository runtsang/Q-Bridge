"""FastBaseEstimator with gradient and shot‑noise support using Qiskit."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.opflow import PauliSumOp

BaseOperator = PauliSumOp  # alias for readability


class FastBaseEstimator:
    """Evaluate expectation values of parameterised circuits.

    The estimator accepts a :class:`qiskit.circuit.QuantumCircuit` and can
    compute expectation values for a list of observables.  It supports
    shot‑noise simulation, caching of statevectors, and analytic gradients
    via the parameter‑shift rule.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        backend: Optional[str] = "statevector_simulator",
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend_name = backend
        self.shots = shots
        self.seed = seed
        self._cache: dict[tuple[float,...], Statevector] = {}
        self._backend = Aer.get_backend(backend)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _statevector(self, circuit: QuantumCircuit) -> Statevector:
        key = tuple(circuit.parameters)
        if key in self._cache:
            return self._cache[key]
        sv = Statevector.from_instruction(circuit)
        self._cache[key] = sv
        return sv

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            circ = self._bind(params)
            if self.shots is not None:
                job = execute(
                    circ,
                    self._backend,
                    shots=self.shots,
                    seed_simulator=self.seed,
                    seed_transpiler=self.seed,
                )
                result = job.result()
                counts = result.get_counts(circ)
                probs = {k: v / self.shots for k, v in counts.items()}
                sv = Statevector.from_label(max(probs, key=probs.get))
                state = sv
            else:
                state = self._statevector(circ)

            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def gradient(
        self,
        observable: BaseOperator,
        parameter_set: Sequence[float],
        *,
        shift: float = np.pi / 2,
    ) -> List[float]:
        """Return the analytic gradient of a single observable.

        Uses the parameter‑shift rule.  The result is a list of partial
        derivatives with respect to each circuit parameter.
        """
        grads = []
        for idx, _ in enumerate(self._parameters):
            shift_plus = self._bind(
                tuple(
                    p + shift if i == idx else p
                    for i, p in enumerate(parameter_set)
                )
            )
            shift_minus = self._bind(
                tuple(
                    p - shift if i == idx else p
                    for i, p in enumerate(parameter_set)
                )
            )
            exp_plus = Statevector.from_instruction(shift_plus).expectation_value(
                observable
            )
            exp_minus = Statevector.from_instruction(shift_minus).expectation_value(
                observable
            )
            grad = (exp_plus - exp_minus) / (2 * np.sin(shift))
            grads.append(float(grad))
        return grads

    def __repr__(self) -> str:
        return (
            f"<FastBaseEstimator circuit={self._circuit.name!r} "
            f"backend={self.backend_name!r} shots={self.shots}>"
        )

__all__ = ["FastBaseEstimator"]
