"""Quantum estimator built on PennyLane QNodes with shot‑noise and gradient support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple

import pennylane as qml
from pennylane import numpy as np

# --------------------------------------------------------------------------- #
# Core estimator
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluate expectation values and gradients for a parametrized quantum circuit.

    Parameters
    ----------
    circuit:
        A PennyLane QNode that returns the expectation value of a single
        observable defined by its measurement in the circuit.
    device:
        Optional PennyLane device name or object; if ``None`` a default.qubit
        device with 1 qubit is used.
    shots:
        Number of shots for stochastic simulation; if ``None`` the device
        default is used.
    """

    def __init__(self, circuit: qml.QNode, device: qml.Device | str | None = None, shots: int | None = None) -> None:
        self.circuit = circuit
        # Resolve device
        if isinstance(device, qml.Device):
            self.device = device
        elif device is None:
            self.device = qml.device("default.qubit", wires=circuit.num_wires, shots=shots)
        else:
            self.device = qml.device(device, wires=circuit.num_wires, shots=shots)

        # Rebuild circuit on the selected device if necessary
        if circuit.device!= self.device:
            self.circuit = qml.QNode(circuit.func, self.device)

    # --------------------------------------------------------------------- #
    # Basic evaluation
    # --------------------------------------------------------------------- #
    def evaluate(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables:
            Iterable of PennyLane operators (e.g., qml.PauliZ(wire)).
        parameter_sets:
            Sequence of parameter vectors.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            # Bind parameters and evaluate
            expectation_values = [self.circuit(*params, observable=obs) for obs in observables]
            results.append(expectation_values)

        return results

    # --------------------------------------------------------------------- #
    # Gradient‑aware evaluation
    # --------------------------------------------------------------------- #
    def evaluate_with_gradients(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> Tuple[List[List[complex]], List[List[List[np.ndarray]]]]:
        """Return expectation values and gradients for each observable.

        Gradients are computed via PennyLane's automatic differentiation
        (parameter‑shift rule). Each gradient is an array matching the shape
        of the parameters passed to the circuit.

        Parameters
        ----------
        observables:
            Iterable of PennyLane operators.
        parameter_sets:
            Sequence of parameter vectors.
        """
        observables = list(observables)
        values: List[List[complex]] = []
        grads: List[List[List[np.ndarray]]] = []

        for params in parameter_sets:
            val_row: List[complex] = []
            grad_row: List[List[np.ndarray]] = []

            for obs in observables:
                # Define a wrapper that fixes the observable
                def wrapped_circuit(*p, observable=obs):
                    return self.circuit(*p, observable=observable)

                # Compute value and gradient
                val = wrapped_circuit(*params)
                grad = qml.grad(wrapped_circuit)(*params)

                val_row.append(val)
                grad_row.append(grad)

            values.append(val_row)
            grads.append(grad_row)

        return values, grads


__all__ = ["FastBaseEstimator"]
