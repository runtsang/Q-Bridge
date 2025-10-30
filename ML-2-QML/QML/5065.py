import pennylane as qml
import numpy as np
from typing import Iterable, List, Sequence, Callable

class HybridFastEstimator:
    """
    Quantum‑centric estimator that evaluates a variational circuit built with
    Pennylane.  The estimator can be configured with an arbitrary circuit
    builder which returns a QNode that, when called, yields a list of
    expectation values for the observables supplied at evaluation time.
    """
    def __init__(self,
                 circuit_builder: Callable[[Sequence[float]], qml.QNode],
                 *,
                 shots: int | None = None,
                 seed: int | None = None):
        """
        Parameters
        ----------
        circuit_builder
            Callable that receives a parameter vector and returns a Pennylane
            QNode.  The QNode must return a list/tuple of expectation values
            that correspond to the observables passed to :meth:`evaluate`.
        shots
            Optional number of shots to emulate measurement noise.
        seed
            Random seed for the shot‑noise generator.
        """
        self.circuit_builder = circuit_builder
        self.shots = shots
        self.rng = np.random.default_rng(seed) if seed is not None else None

    def evaluate(self,
                 observables: Iterable[qml.operation.Operator],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """
        Compute expectation values for each observable and parameter set.

        Parameters
        ----------
        observables
            Sequence of Pennylane operators.  The order of the returned values
            matches this list.
        parameter_sets
            Sequence of parameter vectors matching the circuit's dimensionality.

        Returns
        -------
        List[List[complex]]
            Nested list of expectation values.  The outer list indexes the
            parameter set, the inner list the observable.
        """
        results: List[List[complex]] = []

        for params in parameter_sets:
            qnode = self.circuit_builder(params)
            row = qnode()  # Expect a list/tuple of expectation values
            results.append(list(row))

        if self.shots is not None and self.rng is not None:
            noisy_results: List[List[complex]] = []
            std = max(1e-6, 1 / np.sqrt(self.shots))
            for row in results:
                noisy_row = [complex(self.rng.normal(v.real, std),
                                     self.rng.normal(v.imag, std)) for v in row]
                noisy_results.append(noisy_row)
            return noisy_results

        return results


__all__ = ["HybridFastEstimator"]
