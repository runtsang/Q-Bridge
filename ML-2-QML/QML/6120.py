"""Enhanced FastBaseEstimator using Pennylane for variational circuits with gradients."""

from __future__ import annotations

import pennylane as qml
from collections.abc import Iterable, Sequence
from typing import List, Optional

class FastBaseEstimator:
    """Variational circuit estimator that evaluates expectation values and gradients for parameter sets."""

    def __init__(self, qnode: qml.QNode):
        """
        Parameters
        ----------
        qnode : Pennylane QNode
            Must accept a list of parameters and optional arguments `observable`, `shots`, and `seed`.
        """
        self.qnode = qnode

    def evaluate(
        self,
        observables: Iterable[qml.operation.Operator] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        return_gradients: bool = False,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Evaluate expectation values (and optionally gradients) for each parameter set.

        Parameters
        ----------
        observables : iterable of Pennylane operators. If None, a default identity is used.
        parameter_sets : list of parameter sequences.
        return_gradients : if True, also return gradients of each observable w.r.t parameters.
        shots : if provided, perform Monte‑Carlo sampling with given shot count.
        seed : random seed for sampling.

        Returns
        -------
        List of rows; each row contains expectation values (and optionally gradients flattened).
        """
        if observables is None:
            observables = [qml.expval(qml.identity(0))]
        observables = list(observables)
        if parameter_sets is None:
            raise ValueError("parameter_sets must be provided")

        results: List[List[complex]] = []

        for params in parameter_sets:
            expvals = [
                self.qnode(*params, observable=obs, shots=shots, seed=seed)
                for obs in observables
            ]
            results.append([float(ev) if isinstance(ev, complex) else ev for ev in expvals])

        if return_gradients:
            grads = self._compute_gradients(observables, parameter_sets)
            results = [row + grad for row, grad in zip(results, grads)]

        return results

    def _compute_gradients(
        self,
        observables: List[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Compute gradients of each observable w.r.t parameters using parameter‑shift rule.
        Returns flattened gradient lists.
        """
        grads_list: List[List[float]] = []
        for params in parameter_sets:
            grads_row: List[float] = []
            for obs in observables:
                grad = qml.grad(self.qnode, argnum=0, obs=obs)(*params)
                grads_row.extend(grad.tolist())
            grads_list.append(grads_row)
        return grads_list


__all__ = ["FastBaseEstimator"]
