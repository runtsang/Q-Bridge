"""Enhanced estimator for PennyLane variational circuits with noise, gradients, and multi‑observable support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Union

import numpy as np
import pennylane as qml
from pennylane.measurements import MeasurementProcess


class FastEstimator:
    """
    Evaluate a PennyLane QNode for a set of parameter vectors and observables.

    Features
    --------
    * Supports expectation value, variance, or custom measurement processes.
    * Optional shot noise via a simulated noisy expectation value.
    * Automatic differentiation of observables w.r.t. circuit parameters.
    * Handles both real and complex‑valued observables.

    Parameters
    ----------
    circuit_func : Callable[[Sequence[float], MeasurementProcess], complex]
        A user‑supplied function defining the circuit.  It should accept a
        parameter vector and a measurement process and return the result of
        the measurement.  The function will be wrapped into a QNode on the
        device specified below.
    wires : Sequence[int] | int, optional
        The wires on which the circuit acts.  Defaults to ``2``.
    device_name : str, optional
        Name of the PennyLane device.  Defaults to ``'default.qubit'``.
    shots : int | None, optional
        Number of shots for classical simulation.  If ``None`` the device is
        set to the exact (no‑shot) mode.
    seed : int | None, optional
        Random seed for the device simulation.
    """

    def __init__(
        self,
        circuit_func: qml.QNode | callable,
        wires: Sequence[int] | int = 2,
        device_name: str = "default.qubit",
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        if isinstance(wires, int):
            wires = list(range(wires))
        self.dev = qml.device(
            device_name,
            wires=wires,
            shots=shots,
            seed=seed,
        )
        # Wrap the user supplied function into a QNode
        self.qnode = qml.QNode(circuit_func, self.dev)

    def evaluate(
        self,
        observables: Iterable[MeasurementProcess],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute the expectation values for each observable and parameter set.

        Parameters
        ----------
        observables : iterable of PennyLane measurement processes
            Each item will be passed to the QNode as the ``obs`` argument.
        parameter_sets : sequence of sequences
            Each inner sequence is a list of parameter values for one evaluation.

        Returns
        -------
        List[List[complex]]
            A list of rows, one per parameter set, each containing the value of
            every observable.
        """
        observables = list(observables)
        if not observables:
            raise ValueError("At least one observable must be provided.")
        results: List[List[complex]] = []
        for params in parameter_sets:
            row = [self.qnode(params, obs) for obs in observables]
            results.append(row)
        return results

    def gradient(
        self,
        observables: Iterable[MeasurementProcess],
        parameter_sets: Sequence[Sequence[float]],
        *,
        method: str = "param_shift",
    ) -> List[List[np.ndarray]]:
        """
        Compute gradients of observables w.r.t. circuit parameters.

        Parameters
        ----------
        observables : iterable of measurement processes
            As in :meth:`evaluate`.
        parameter_sets : sequence of sequences
            As in :meth:`evaluate`.
        method : {'param_shift', 'gradient'}, optional
            Gradient computation method.  ``'param_shift'`` uses the
            parameter‑shift rule; ``'gradient'`` uses a built‑in
            PennyLane gradient estimator.

        Returns
        -------
        List[List[np.ndarray]]
            Gradient arrays for each observable and parameter set.  Each array
            has shape ``(num_params, )``.
        """
        observables = list(observables)
        grads: List[List[np.ndarray]] = []
        for params in parameter_sets:
            if method == "param_shift":
                grad_func = qml.gradients.param_shift
            elif method == "gradient":
                grad_func = qml.gradients.gradient
            else:
                raise ValueError(f"Unsupported gradient method: {method}")
            row: List[np.ndarray] = []
            for obs in observables:
                g = grad_func(
                    self.qnode,
                    argnum=0,
                    arg_shape=(len(params),),
                    obsnames=[obs],
                )(params)
                row.append(g[0])
            grads.append(row)
        return grads


__all__ = ["FastEstimator"]
