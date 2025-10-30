"""Quantum estimator that evaluates expectation values of PennyLane circuits."""

from __future__ import annotations

import pennylane as qml
from typing import Iterable, List, Sequence

class FastEstimator:
    """Hybrid estimator for PennyLane circuits.

    Parameters
    ----------
    circuit : qml.QNode
        A PennyLane QNode that accepts a parameter vector and an observable.
    """
    def __init__(self, circuit: qml.QNode) -> None:
        self.circuit = circuit

    def evaluate(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : Iterable[qml.operation.Operator]
            List of quantum operators.
        parameter_sets : Sequence[Sequence[float]]
            Sequences of parameter vectors.
        shots : int | None
            Number of shots to simulate.  If ``None`` exact values are used.
        seed : int | None
            Random seed used for shot noise.

        Returns
        -------
        List[List[float]]
            Row‑first list of expectation values.
        """
        import numpy as np

        results: List[List[float]] = []

        for params in parameter_sets:
            row: List[float] = []
            for obs in observables:
                if shots is not None:
                    dev = qml.device("default.qubit", wires=obs.wires, shots=shots)
                    qnode = qml.QNode(self.circuit.func, dev, interface="autograd")
                    val = qnode(params, obs)
                else:
                    val = self.circuit(params, obs)
                row.append(float(val))
            results.append(row)

        if shots is not None and seed is not None:
            rng = np.random.default_rng(seed)
            noisy_results: List[List[float]] = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy_results.append(noisy_row)
            return noisy_results

        return results

__all__ = ["FastEstimator"]
