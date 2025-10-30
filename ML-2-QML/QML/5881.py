import pennylane as qml
import numpy as np
from pennylane import matrix as qml_matrix
from collections.abc import Iterable, Sequence
from typing import List, Callable

class FastBaseEstimatorQuantum:
    """Quantum estimator that evaluates expectation values of parameterized circuits."""
    def __init__(self, qnode: qml.QNode, shots: int = 8192) -> None:
        """
        Parameters
        ----------
        qnode:
            A Pennylane QNode that returns a statevector when called with parameters.
        shots:
            Number of shots for stochastic simulation (unused with statevector but kept for API compatibility).
        """
        self.qnode = qnode
        self.shots = shots

    def evaluate(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]]
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for params in parameter_sets:
            state = self.qnode(*params)  # statevector
            row = [np.vdot(state, qml_matrix(obs) @ state) for obs in observables]
            results.append(row)
        return results

    def gradient(
        self,
        observable: qml.operation.Operator,
        parameter_sets: Sequence[Sequence[float]]
    ) -> List[List[float]]:
        """Return gradients of the expectation value of a single observable
        with respect to all circuit parameters for each parameter set."""
        grads: List[List[float]] = []

        def exp_fn(*params):
            state = self.qnode(*params)
            return np.vdot(state, qml_matrix(observable) @ state)

        grad_fn = qml.grad(exp_fn)

        for params in parameter_sets:
            grad = grad_fn(*params)
            grads.append(list(grad))
        return grads


__all__ = ["FastBaseEstimatorQuantum"]
