"""Hybrid estimator that evaluates a Pennylane QNode.

The class shares the name `HybridBaseEstimator` and provides the same
`evaluate` API.  It accepts a QNode and a list of measurement
functions, optionally performing shot noise when a shot number is
provided.  When ``shots`` is ``None`` a state‑vector simulation is
used, otherwise a device with the specified shot count is employed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

# Define a simple protocol for observables
ObservableFunc = Callable[[np.ndarray], complex]


class HybridBaseEstimator:
    """Quantum estimator based on a Pennylane QNode.

    Parameters
    ----------
    qnode : qml.QNode
        Callable that returns a statevector or measurement.
    wires : Sequence[int] | int
        Number of wires for the default qubit device.
    shots : int | None
        If ``None`` the default state‑vector simulator is used.
        Otherwise a shot‑counted device is created.
    """

    def __init__(
        self,
        qnode: qml.QNode,
        wires: int | Sequence[int] = 1,
        shots: Optional[int] = None,
    ) -> None:
        self.qnode = qnode
        self.shots = shots
        self.device = qml.device(
            "default.qubit",
            wires=wires,
            shots=None if shots is None else shots,
        )

    def evaluate(
        self,
        observables: Iterable[ObservableFunc],
        parameter_sets: Sequence[Sequence[float]],
        *,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """Evaluate expectation values for each parameter set.

        Parameters
        ----------
        observables : iterable
            Functions that map a statevector (numpy array) to a
            complex number.  Typical usage is ``lambda sv: sv.conj().T @ O @ sv``.
        parameter_sets : sequence
            Iterable of parameter vectors.
        seed : int, optional
            Random seed to reproducible shot noise.

        Returns
        -------
        List[List[complex]]
            Nested list with shape
            ``(len(parameter_sets), len(observables))``.
        """
        observables = list(observables) or [lambda sv: 1.0]
        results: List[List[complex]] = []

        rng = np.random.default_rng(seed)

        for params in parameter_sets:
            # Run the QNode to obtain either a statevector or a measurement
            # (the QNode is executed on the device defined in __init__).
            result = self.qnode(*params)

            # If the node returns a statevector (array of complex numbers),
            # use it directly.  Otherwise treat the result as the expectation
            # of the identity operator.
            if isinstance(result, np.ndarray):
                state = result
            else:
                state = np.array([result.real + 1j * result.imag])

            row: List[complex] = []
            for observable in observables:
                try:
                    value = observable(state)
                except Exception:
                    # Fallback: if the observable is a Pennylane operator,
                    # compute the quadratic form manually.
                    op = observable.data if hasattr(observable, "data") else observable
                    value = state.conj().T @ op @ state
                row.append(value)

            # If shots are requested, perturb the expectation with Gaussian noise.
            if self.shots is not None:
                noisy_row = []
                for v in row:
                    if isinstance(v, complex):
                        noise = (
                            rng.normal(0, 1 / np.sqrt(self.shots))
                            + 1j * rng.normal(0, 1 / np.sqrt(self.shots))
                        )
                    else:
                        noise = rng.normal(0, 1 / np.sqrt(self.shots))
                    noisy_row.append(v + noise)
                row = noisy_row

            results.append(row)

        return results


# Backwards‑compatibility alias
FastBaseEstimator = HybridBaseEstimator
__all__ = ["HybridBaseEstimator", "FastBaseEstimator"]
