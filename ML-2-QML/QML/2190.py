"""Quantum variant of FastBaseEstimator using Pennylane, supporting analytic gradients and shot‑noise simulation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Callable

import pennylane as qml
import pennylane.numpy as pnp

ScalarObservable = Callable[[pnp.ndarray], pnp.ndarray | float]


def _ensure_batch(values: Sequence[float]) -> pnp.ndarray:
    """Convert a sequence into a 2‑D array for batching."""
    arr = pnp.array(values, dtype=pnp.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


class FastBaseEstimator:
    """Evaluate a Pennylane QNode for many parameter sets and observables.

    Parameters
    ----------
    circuit : Callable[..., pnp.ndarray]
        Variational circuit that accepts raw parameters and returns a state
        or a single expectation value.
    device : qml.Device
        Pennylane device backend.  It should expose ``wires`` and ``shots``.
    observables : Iterable[qml.operation.Operator] | None
        List of operators to measure.  If ``None`` a default Pauli‑Z is used.
    """

    def __init__(
        self,
        circuit: Callable[..., pnp.ndarray],
        device: qml.Device,
        observables: Iterable[qml.operation.Operator] | None = None,
    ) -> None:
        self._circuit = circuit
        self.device = device
        self.observables = list(observables or [qml.PauliZ(0)])
        self._qnodes = [
            qml.QNode(
                lambda *params, obs=obs: self._apply(obs, *params),
                device,
                interface="autograd",
            )
            for obs in self.observables
        ]

    def _apply(self, observable: qml.operation.Operator, *params: float) -> float:
        self._circuit(*params)
        return qml.expval(observable)

    def evaluate(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Return expectation values for each parameter set and observable."""
        # Use internally stored observables; the argument is ignored for backward
        # compatibility with the original API.
        results: List[List[complex]] = []
        for params in parameter_sets:
            row = [qnode(*params).item() for qnode in self._qnodes]
            results.append(row)
        return results

    def evaluate_with_gradient(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
        grad_shifts: int | None = None,
    ) -> List[List[complex]]:
        """Return expectation values and analytic gradients using the parameter‑shift rule.

        Parameters
        ----------
        grad_shifts
            Number of parameter‑shift evaluations per gradient.  ``1`` is the
            minimal default; larger values can reduce variance for complex
            observables.
        """
        grad_shifts = grad_shifts or 1
        results: List[List[complex]] = []
        for params in parameter_sets:
            row = [qnode(*params).item() for qnode in self._qnodes]
            grads = []
            for idx in range(len(params)):
                def shift_fn(p):
                    shifted = list(params)
                    shifted[idx] += p
                    return self.evaluate(observables, [shifted])[0][0]  # single observable

                grad = qml.gradients.param_shift(shift_fn, shift=grad_shifts)(0.0)
                grads.append(grad)
            results.append(row + grads)
        return results


__all__ = ["FastBaseEstimator"]
