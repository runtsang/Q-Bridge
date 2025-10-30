"""Enhanced quantum estimator with Pennylane integration, parameter‑shift gradients, and shot‑noise simulation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Union

import pennylane as qml
import numpy as np

# Define a type that accepts either Pennylane operators or custom callable observables
QuantumObservable = Union[qml.operation.Operator, Callable[[qml.State], complex]]


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized Pennylane circuit.

    The estimator supports a user‑supplied circuit builder that receives a sequence of
    parameters and returns a Pennylane circuit (or a QNode).  It also offers
    parameter‑shift gradients and optional shot‑noise simulation.

    Parameters
    ----------
    circuit_builder : Callable[[Sequence[float]], qml.QNode]
        A function that builds a Pennylane QNode given a parameter vector.
    device : qml.Device, optional
        Pennylane device; defaults to ``default.qubit``.
    shots : int, optional
        Number of shots for simulation; if ``None`` the device runs in state‑vector mode.
    """

    def __init__(
        self,
        circuit_builder: Callable[[Sequence[float]], qml.QNode],
        device: qml.Device | None = None,
        shots: int | None = None,
    ) -> None:
        self.circuit_builder = circuit_builder
        self.device = device or qml.device("default.qubit", wires=0, shots=shots)
        self.shots = shots

    def _build_qnode(self, observables: Iterable[QuantumObservable]) -> qml.QNode:
        """Create a QNode that returns expectation values for the given observables."""
        obs_list = list(observables)

        @qml.qnode(self.device, interface="torch", diff_method="parameter-shift")
        def circuit(*params):
            # Build the circuit with the supplied parameters
            self.circuit_builder(*params)
            # Return expectation values for all observables
            return [qml.expval(obs) for obs in obs_list]

        return circuit

    def evaluate(
        self,
        observables: Iterable[QuantumObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of Pennylane operators or callables that take a state vector.
        parameter_sets
            Sequence of parameter vectors to evaluate.

        Returns
        -------
        List[List[complex]]
            Outer list over parameter sets, inner list over observables.
        """
        circuit = self._build_qnode(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            out = circuit(*params)
            results.append([complex(val) for val in out])

        return results

    def compute_gradients(
        self,
        observables: Iterable[QuantumObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[List[float]]]:
        """Compute parameter‑shift gradients of each observable w.r.t. parameters.

        Returns a nested list:
            [parameter_set][observable][parameter].
        """
        circuit = self._build_qnode(observables)
        grad_circuit = qml.gradients.param_shift(circuit)
        grads: List[List[List[float]]] = []

        for params in parameter_sets:
            grad_vals = grad_circuit(*params)  # shape: (num_obs, num_params)
            grads.append([list(g) for g in grad_vals])

        return grads

    def evaluate_with_shots(
        self,
        observables: Iterable[QuantumObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Simulate shot noise by running the circuit on a shot‑based device.

        Parameters
        ----------
        shots
            Number of shots for each evaluation.
        seed
            Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)
        # Temporarily replace the device's shot count
        original_shots = self.device.shots
        self.device.shots = shots

        circuit = self._build_qnode(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            out = circuit(*params)
            # Pennylane returns a list of expectation values; convert to complex
            results.append([complex(val) for val in out])

        # Restore original shot count
        self.device.shots = original_shots
        return results


__all__ = ["FastBaseEstimator"]
