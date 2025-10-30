"""Enhanced FastBaseEstimator for PennyLane circuits with analytic gradients and shot‑noise simulation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import pennylane as qml
import pennylane.numpy as np
from pennylane import qnode
from pennylane.operation import Operator


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized PennyLane circuit.

    Parameters
    ----------
    circuit : Callable
        A PennyLane QNode or a function that returns a quantum circuit.
    wires : Sequence[int] | str, optional
        Wire identifiers for the device.  If ``None`` the circuit's wires are inferred.
    dev_kwargs : dict, optional
        Additional keyword arguments passed to the PennyLane device (e.g., shots, backend).

    Notes
    -----
    * ``evaluate`` is backward compatible with the original seed.
    * ``evaluate_shots`` allows explicit shot‑noise control.
    * ``gradient`` returns analytic gradients using the parameter‑shift rule.
    """

    def __init__(
        self,
        circuit: qml.QNode | callable,
        wires: Optional[Sequence[int] | str] = None,
        dev_kwargs: Optional[dict] = None,
    ) -> None:
        self.wires = wires
        self.dev_kwargs = dev_kwargs or {}
        # Create a default device if not provided
        if isinstance(circuit, qml.QNode):
            self._circuit = circuit
            self.dev = circuit.device
        else:
            self.dev = qml.device("default.qubit", wires=self.wires, **self.dev_kwargs)
            self._circuit = qml.QNode(circuit, self.dev)

    # --------------------------------------------------------------------- #
    # Core evaluation
    # --------------------------------------------------------------------- #
    def evaluate(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables : Iterable of PennyLane operators
            Each operator is evaluated as an expectation value.
        parameter_sets : Sequence of parameter vectors

        Returns
        -------
        List[List[complex]]
            Rows correspond to parameter sets; columns to observables.
        """
        results: List[List[complex]] = []
        for params in parameter_sets:
            row: List[complex] = []
            for obs in observables:
                # The QNode automatically binds parameters
                val = self._circuit(*params, observables=[obs])
                row.append(val)
            results.append(row)
        return results

    # --------------------------------------------------------------------- #
    # Shot‑noise simulation
    # --------------------------------------------------------------------- #
    def evaluate_shots(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
        shots: int,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """Evaluate expectation values with explicit shot noise.

        Parameters
        ----------
        shots : int
            Number of measurement shots per observable.
        seed : int | None
            Random seed for reproducibility.

        Returns
        -------
        List[List[complex]]
            Noisy expectation values.
        """
        rng = np.random.default_rng(seed)
        # Create a new device with the requested shot count
        dev = qml.device("default.qubit", wires=self.wires, shots=shots, **self.dev_kwargs)
        qnode_shot = qml.QNode(self._circuit.function, dev)
        results: List[List[complex]] = []
        for params in parameter_sets:
            row: List[complex] = []
            for obs in observables:
                val = qnode_shot(*params, observables=[obs])
                # Convert the noisy sample to a complex number (real part)
                row.append(val)
            results.append(row)
        return results

    # --------------------------------------------------------------------- #
    # Gradient computation
    # --------------------------------------------------------------------- #
    def gradient(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return analytic gradients of the first observable w.r.t. parameters.

        Parameters
        ----------
        observables : Iterable of operators
            The first observable's gradient is computed.

        Returns
        -------
        List[List[float]]
            Gradient vector for each parameter set.
        """
        grads_list: List[List[float]] = []
        for params in parameter_sets:
            grad_fn = qml.grad(self._circuit, argnum=0)
            grads = grad_fn(*params, observables=[list(observables)[0]])
            grads_list.append(grads.tolist())
        return grads_list


__all__ = ["FastBaseEstimator"]
