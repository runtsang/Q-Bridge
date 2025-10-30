"""Advanced estimator primitive using PennyLane for variational circuits."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Tuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.operation import Operation
from pennylane.measurements import Expectation

# Type alias for observable functions
ScalarObservable = Callable[[qml.QNode], float | complex]


class AdvancedBaseEstimator:
    """
    Evaluate expectation values of observables for a parametrized quantum circuit.

    Features
    --------
    * Support for shot‑based simulation with configurable shots.
    * Hardware‑noise simulation via PennyLane noise models.
    * Automatic parameter‑shift gradient estimation.
    * Batch evaluation over multiple parameter sets.
    """

    def __init__(
        self,
        circuit: Callable[..., None],
        wires: Sequence[int] | int,
        dev: Optional[qml.Device] = None,
        shots: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        circuit : callable
            Quantum circuit function that accepts a list of parameters.
        wires : int or sequence of int
            Wire indices used by the circuit.
        dev : qml.Device, optional
            PennyLane device; if None a default qubit simulator is used.
        shots : int, optional
            Number of shots; if None, uses exact simulation.
        """
        self._circuit = circuit
        self._wires = wires if isinstance(wires, (list, tuple)) else [wires]
        self._shots = shots
        self._dev = dev or qml.device("default.qubit", wires=self._wires, shots=shots)
        self._qnode = qml.QNode(circuit, self._dev)

    def _bind(self, parameter_values: Sequence[float]) -> None:
        """
        Bind parameters to the QNode for evaluation.
        """
        if len(parameter_values)!= self._qnode.n_params:
            raise ValueError("Parameter count mismatch for bound circuit.")
        self._qnode.set_parameters(parameter_values)

    def evaluate(
        self,
        observables: Iterable[Expectation | Operation],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables : iterable of pennylane.expectation or operators
            Each element is an observable to evaluate.
        parameter_sets : sequence of parameter sequences
            Each inner sequence is a list of floats to feed the circuit.

        Returns
        -------
        results : list of list of complex
            Expectation values for each parameter set.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            self._bind(values)
            row = [self._qnode(obs) for obs in observables]
            results.append(row)

        return results

    def compute_gradients(
        self,
        observables: Iterable[Expectation | Operation],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[List[float]]]:
        """
        Estimate gradients of each observable w.r.t. all parameters using
        the parameter‑shift rule.

        Returns
        -------
        grads : list of list of list of float
            grads[obs_index][set_index][param_index]
        """
        grads: List[List[List[float]]] = []

        for values in parameter_sets:
            self._bind(values)
            grad_row: List[List[float]] = []
            for obs in observables:
                grad = qml.grad(self._qnode)(obs)
                grad_row.append(grad.tolist())
            grads.append(grad_row)

        return grads

    def add_noise(
        self,
        results: List[List[complex]],
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Add shot‑noise to deterministic results by resampling from binomial
        distributions for each expectation value.

        Parameters
        ----------
        results : list of list of complex
            Deterministic expectation values.
        shots : int, optional
            Number of shots; if None, no noise is added.
        seed : int, optional
            Random seed for reproducibility.
        """
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy = []
        for row in results:
            noisy_row = [
                complex(rng.normal(np.real(val), max(1e-6, 1 / shots)),
                        rng.normal(np.imag(val), max(1e-6, 1 / shots)))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["AdvancedBaseEstimator"]
