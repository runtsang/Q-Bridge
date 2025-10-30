import pennylane as qml
import pennylane.numpy as np
from collections.abc import Iterable, Sequence
from typing import List, Optional

class FastBaseEstimator:
    """
    Lightweight estimator for variational quantum circuits that supports multiple
    backends, shot‑based or analytic expectation evaluation, and gradient
    computation via parameter‑shift or automatic differentiation.
    """

    def __init__(
        self,
        circuit: qml.QNode,
        dev: qml.Device,
        shots: int | None = None,
    ) -> None:
        self.circuit = circuit
        self.dev = dev
        self.shots = shots

    def _bind(self, parameter_values: Sequence[float]) -> qml.QNode:
        """Return a circuit with the parameters bound to the supplied values."""
        def bound_circuit(*args, **kwargs):
            return self.circuit(*parameter_values, *args, **kwargs)
        return qml.QNode(bound_circuit, self.dev)

    def evaluate(
        self,
        observables: Iterable[qml.operation.Operation],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for every parameter set and observable.
        Uses analytic expectation if shots is None, otherwise performs sampling.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound_circuit = self._bind(values)
            row: List[complex] = []
            for obs in observables:
                if self.shots is None:
                    exp_val = qml.expval(obs)(bound_circuit)
                else:
                    exp_val = qml.expval(obs)(bound_circuit, shots=self.shots)
                row.append(exp_val)
            results.append(row)
        return results

    def compute_gradients(
        self,
        observables: Iterable[qml.operation.Operation],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[List[complex]]]:
        """
        Return the gradient of each observable with respect to the circuit parameters.
        Uses the parameter‑shift rule for analytic gradients and the adjoint
        algorithm for shot‑based evaluation when available.
        """
        observables = list(observables)
        grads: List[List[List[complex]]] = []
        for values in parameter_sets:
            row_grads: List[List[complex]] = []
            for obs in observables:
                if self.shots is None:
                    grad = qml.gradients.param_shift(
                        lambda *p: qml.expval(obs)(self.circuit, *p)
                    )(values)
                else:
                    grad = qml.gradients.adjoint(
                        lambda *p: qml.expval(obs)(self.circuit, *p)
                    )(values)
                row_grads.append(grad)
            grads.append(row_grads)
        return grads
