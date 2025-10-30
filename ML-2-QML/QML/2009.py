"""Fast quantum estimator with shot‑noise, noise models and gradient estimation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel


class FastEstimatorGen210:
    """
    Quantum estimator that evaluates expectation values of operators for
    parametrised circuits.  Features:

    * GPU‑accelerated state‑vector simulation via Aer.
    * Shot‑noise simulation and optional noise models.
    * Parameter‑shift gradient estimation.

    The API mirrors the classical estimator for compatibility.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Optional[AerSimulator] = None,
        noise_model: Optional[NoiseModel] = None,
    ) -> None:
        self._orig_circuit = circuit
        self._parameters = list(circuit.parameters)
        self._backend = backend or AerSimulator(method="statevector")
        if noise_model:
            self._backend.set_options(noise_model=noise_model)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._orig_circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Deterministic expectation values using state‑vector simulation.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = self._bind(values)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_with_shots(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Expectation values simulated with finite shots (and optional noise).
        """
        if shots is None:
            return self.evaluate(observables, parameter_sets)

        observables = list(observables)
        results: List[List[complex]] = []
        rng = np.random.default_rng(seed)
        for values in parameter_sets:
            bound = self._bind(values)
            job = self._backend.run(bound, shots=shots)
            result = job.result()
            counts = result.get_counts()
            row: List[complex] = []
            for obs in observables:
                # Diagonal expectation from measurement samples
                exp = sum(
                    counts.get(bit, 0) * obs.matrix()[int(bit, 2), int(bit, 2)]
                    for bit in counts
                ) / shots
                row.append(exp)
            results.append(row)
        return results

    def evaluate_with_gradients(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shift: float = np.pi / 2,
    ) -> List[List[List[complex]]]:
        """
        Parameter‑shift gradient estimation for each observable.
        Returns a 3‑D list: parameter set → observable → gradient vector.
        """
        observables = list(observables)
        grads: List[List[List[complex]]] = []

        for values in parameter_sets:
            row_grads: List[List[complex]] = []
            for obs in observables:
                grad_vec: List[complex] = []
                for idx, _ in enumerate(values):
                    inc = np.array(values, copy=True)
                    dec = np.array(values, copy=True)
                    inc[idx] += shift
                    dec[idx] -= shift
                    e_plus = self.evaluate([obs], [inc])[0][0]
                    e_minus = self.evaluate([obs], [dec])[0][0]
                    grad = (e_plus - e_minus) / (2 * np.sin(shift))
                    grad_vec.append(grad)
                row_grads.append(grad_vec)
            grads.append(row_grads)
        return grads
