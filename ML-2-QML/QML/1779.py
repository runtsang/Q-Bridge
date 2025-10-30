"""FastBaseEstimatorPro for PennyLane circuits.

Features:
  * Batched evaluation of multiple parameter sets.
  * Shot‑noise simulation via the `shots` argument.
  * Analytic gradient norms for each observable with respect to the circuit parameters.
  * Device abstraction: CPU or GPU simulation.
"""

from __future__ import annotations

import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import pennylane as qml


class FastBaseEstimatorPro:
    """Estimator for PennyLane quantum circuits."""

    def __init__(
        self,
        circuit_builder: Callable[[Sequence[float]], qml.tape.QuantumTape],
        device: Optional[qml.Device] = None,
    ) -> None:
        """
        Parameters
        ----------
        circuit_builder : Callable[[Sequence[float]], qml.tape.QuantumTape]
            Function that takes a parameter vector and returns a quantum tape
            describing the circuit (without measurements).
        device : Optional[qml.Device]
            PennyLane device to run the circuit on. If None, defaults to
            the CPU simulator ``default.qubit``.
        """
        self._circuit_builder = circuit_builder
        self._device = device or qml.device("default.qubit", wires=1)

    def _build_tape(self, params: Sequence[float]) -> qml.tape.QuantumTape:
        """Construct a tape with the supplied parameters."""
        return self._circuit_builder(params)

    def evaluate(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : Iterable[qml.operation.Operator]
            Pennylane operators whose expectation values are desired.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors for the circuit.
        shots : Optional[int]
            Number of measurement shots for finite‑shot simulation. If None,
            the exact expectation value is returned.

        Returns
        -------
        List[List[complex]]
            For each parameter set, a list containing the expectation values.
        """
        results: List[List[complex]] = []

        for params in parameter_sets:
            tapes = []
            for obs in observables:
                tape = self._build_tape(params)
                tape.measurements.append(obs)
                tapes.append(tape)

            # Execute all tapes in a single batch
            batch = qml.execute(
                tapes,
                self._device,
                gradient_fn=None,
                shots=shots,
            )
            results.append(batch)

        return results

    def gradients(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Return the Euclidean norm of the gradient of each observable with respect
        to the circuit parameters for every parameter set.

        Parameters
        ----------
        observables : Iterable[qml.operation.Operator]
            Pennylane operators.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors for the circuit.

        Returns
        -------
        List[List[float]]
            For each parameter set, a list containing the gradient norms.
        """
        grad_results: List[List[float]] = []

        for params in parameter_sets:
            tape = self._build_tape(params)
            # Mark all parameters as trainable
            tape.trainable_params = range(len(params))
            # Compute analytic gradients for each observable
            grad_tapes, grad_vals = qml.gradients.gradient_function(tape, device=self._device)
            row: List[float] = []
            for g in grad_vals:
                # Each g is a list of arrays, one per trainable parameter
                flat = np.concatenate([arr.flatten() for arr in g])
                row.append(float(np.linalg.norm(flat)))
            grad_results.append(row)

        return grad_results


__all__ = ["FastBaseEstimatorPro"]
