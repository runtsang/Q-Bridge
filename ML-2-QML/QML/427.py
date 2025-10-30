"""Advanced estimator primitive using Qiskit with shot noise and gradient support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Sequence as Seq, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit import Parameter
from qiskit.opflow import PauliSumOp
from qiskit.providers.aer import AerSimulator
from qiskit.opflow import StateFn, ExpectationFactory, AerPauliExpectation

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit.

    The estimator can perform exact state‑vector evaluation or
    sample‑based estimation via AerSimulator.  It also exposes a
    parameter‑shift gradient routine.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._simulator = AerSimulator()
        self._expectation = AerPauliExpectation().convert

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Operator | PauliSumOp],
        parameter_sets: Sequence[Seq[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of Qiskit operators (Operator or PauliSumOp) to evaluate.
        parameter_sets
            Sequence of parameter vectors.
        shots
            If provided, perform Monte‑Carlo sampling; otherwise use
            state‑vector evaluation.
        seed
            Random seed for reproducible sampling.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound = self._bind(values)
            if shots is None:
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                job = self._simulator.run(bound, shots=shots, seed_simulator=seed)
                result = job.result()
                counts = result.get_counts(bound)
                # Convert counts into a statevector for expectation evaluation
                state = Statevector.from_counts(bound, counts)
                row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def gradient(
        self,
        observables: Iterable[Operator | PauliSumOp],
        parameter_sets: Sequence[Seq[float]],
    ) -> List[List[List[float]]]:
        """
        Compute gradients of observables w.r.t. circuit parameters using
        the parameter‑shift rule.  Returns a list of rows; each row
        contains a list of gradient vectors (one per observable) for the
        corresponding parameter set.
        """
        observables = list(observables)
        grads: List[List[List[float]]] = []
        shift = np.pi / 2

        for values in parameter_sets:
            row: List[List[float]] = []
            for obs in observables:
                grad_vec: List[float] = []
                for i, _ in enumerate(self._parameters):
                    pos = list(values); pos[i] += shift
                    neg = list(values); neg[i] -= shift
                    val_pos = self.evaluate([obs], [pos], shots=None)[0][0]
                    val_neg = self.evaluate([obs], [neg], shots=None)[0][0]
                    grad = 0.5 * (val_pos - val_neg)
                    grad_vec.append(float(grad))
                row.append(grad_vec)
            grads.append(row)
        return grads


__all__ = ["FastBaseEstimator"]
