"""Combined quantum estimator that unifies Qiskit circuits, quantum modules, and classical wrappers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_aer import AerSimulator

# Define a generic type that can be a Qiskit circuit or a callable that returns a statevector
QuantumModel = Union[QuantumCircuit, Callable[[Sequence[float]], Statevector]]


class FastCombinedQuantumEstimator:
    """
    Evaluate expectation values of observables for a collection of quantum models.
    Supports deterministic statevector evaluation and shot‑based sampling.
    """

    def __init__(self, models: Sequence[QuantumModel]) -> None:
        """
        :param models: sequence of Qiskit QuantumCircuit objects or callables returning Statevector.
        """
        self.models = list(models)
        self.simulator = AerSimulator(method="statevector")

    def _bind(self, circuit: QuantumCircuit, params: Sequence[float]) -> QuantumCircuit:
        if len(params)!= len(circuit.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(circuit.parameters, params))
        return circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[List[complex]]]:
        """
        Return a 3‑D list: outer over models, middle over parameter sets, inner over observables.
        """
        observables = list(observables)
        rng = np.random.default_rng(seed)

        results: List[List[List[complex]]] = []

        for model in self.models:
            raw_rows: List[List[complex]] = []

            for params in parameter_sets:
                # obtain statevector
                if isinstance(model, QuantumCircuit):
                    circuit = self._bind(model, params)
                    state = Statevector.from_instruction(circuit)
                else:  # callable
                    state = model(params)

                row = [state.expectation_value(obs) for obs in observables]
                raw_rows.append(row)

            if shots is None:
                results.append(raw_rows)
                continue

            # shot sampling
            noisy_rows: List[List[complex]] = []
            for row in raw_rows:
                noisy_row = [
                    complex(
                        rng.normal(np.real(mean), max(1e-6, 1 / shots)),
                        rng.normal(np.imag(mean), max(1e-6, 1 / shots)),
                    )
                    for mean in row
                ]
                noisy_rows.append(noisy_row)
            results.append(noisy_rows)

        return results


__all__ = ["FastCombinedQuantumEstimator"]
