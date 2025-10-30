"""Advanced base estimator for quantum circuits with batched evaluation and parameter‑shift gradients."""

from __future__ import annotations

import pennylane as qml
from typing import Iterable, List, Sequence

class AdvancedBaseEstimator:
    """Evaluate expectation values and gradients for parametrized quantum circuits."""

    def __init__(
        self,
        circuit: qml.QNode,
        device_name: str = "default.qubit",
        shots: int | None = None,
    ) -> None:
        # Use the circuit's device if available, otherwise create a new one.
        self.circuit = circuit
        self.device = (
            circuit.device
            if hasattr(circuit, "device")
            else qml.device(device_name, wires=circuit.wires, shots=shots)
        )
        self._parameters = circuit.parameters

    def evaluate(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable.

        The provided circuit must return a list of expectation values in the same
        order as the ``observables`` iterable.
        """
        results: List[List[complex]] = []
        for params in parameter_sets:
            outputs = self.circuit(*params)
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            results.append(outputs)
        return results

    def _observable_function(self, idx: int):
        """Return a QNode that evaluates the expectation value of the observable at index ``idx``."""
        @qml.qnode(self.device)
        def f(*params):
            outputs = self.circuit(*params)
            return outputs[idx]
        return f

    def gradient(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[List[float]]]:
        """Return parameter‑shift gradients for each observable.

        The returned structure is a list over parameter sets, each containing a list
        over observables, each containing a gradient vector (list of floats).
        """
        grads: List[List[List[float]]] = []
        for params in parameter_sets:
            param_grads: List[List[float]] = []
            for idx, _ in enumerate(observables):
                f = self._observable_function(idx)
                grad = qml.grad(f, argnum=0)(*params)
                param_grads.append(grad)
            grads.append(param_grads)
        return grads


__all__ = ["AdvancedBaseEstimator"]
