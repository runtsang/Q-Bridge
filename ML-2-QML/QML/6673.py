"""Quantum estimator with shot‑noise simulation and gradient via parameter‑shift.

The `FastEstimator` class wraps a parametrised `QuantumCircuit` and
provides:
* batched expectation evaluation,
* optional shot‑noise using Qiskit's Aer simulator,
* analytic gradients computed with the parameter‑shift rule,
* vectorised measurement of multiple observables.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.opflow import PauliSumOp, StateFn, ExpectationFactory, Gradient
from qiskit.opflow.gradients import ParameterShift

class FastEstimator:
    """Estimator for a parametrised quantum circuit with noise and gradients.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to evaluate.  All parameters must be `Parameter` objects.
    backend : str or Aer backend
        Backend for simulation.  Defaults to Aer.get_backend('statevector_simulator').
    """

    def __init__(self, circuit: QuantumCircuit, backend=None) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend = backend or Aer.get_backend("statevector_simulator")

    # ------------------------------------------------------------------ public API
    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Operator | PauliSumOp],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound = self._bind(values)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        return results

    def evaluate_shots(
        self,
        observables: Iterable[Operator | PauliSumOp],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int = 1024,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Same as :meth:`evaluate` but samples expectation values using shots.

        Parameters
        ----------
        shots : int
            Number of shots for each circuit execution.
        seed : int | None
            RNG seed for reproducibility.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound = self._bind(values)
            job = execute(
                bound,
                backend=Aer.get_backend("qasm_simulator"),
                shots=shots,
                seed_simulator=seed,
            )
            counts = job.result().get_counts()
            probs = {k: v / shots for k, v in counts.items()}

            # Build a state vector from the probability distribution
            probs_vec = np.array([probs.get(f"{int(b, 2):0{bound.num_qubits}b}", 0.0)
                                 for b in range(2 ** bound.num_qubits)])
            state = Statevector(probs_vec)

            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        return results

    def gradients(
        self,
        observables: Iterable[Operator | PauliSumOp],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[Tuple[complex,...]]]:
        """Return analytic gradients for each observable w.r.t. each parameter.

        Uses the parameter‑shift rule implemented by Qiskit's OpFlow.
        """
        observables = list(observables)
        grads: List[List[Tuple[complex,...]]] = []

        for values in parameter_sets:
            bound = self._bind(values)
            # Build a gradient operator for all parameters
            grad_ops = [ParameterShift(obs) for obs in observables]
            grad_results: List[Tuple[complex,...]] = []

            for grad_op in grad_ops:
                # grad_op expects a function that maps params -> expectation
                def func(params):
                    circ = self._bind(params)
                    state = Statevector.from_instruction(circ)
                    return state.expectation_value(grad_op.observable)

                # Evaluate gradient at current parameter values
                grad = grad_op.gradient(func, self._parameters, values)
                grad_results.append(tuple(grad))

            grads.append(grad_results)

        return grads


__all__ = ["FastEstimator"]
