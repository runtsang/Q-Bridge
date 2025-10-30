"""Quantum estimator built with PennyLane.

Supports:
* Variational circuit evaluation on CPU or GPU.
* Multiple observables per circuit.
* Optional shot‑noise simulation.
* Compatibility with the original API.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import pennylane as qml
from pennylane import numpy as pnp


class FastBaseEstimator:
    """Evaluate a PennyLane variational circuit for multiple parameter sets.

    Parameters
    ----------
    circuit:
        A function that takes a 1‑D array of parameters and returns a list of
        expectation values for the observables provided at construction.
    observables:
        Iterable of PennyLane observables (e.g. qml.PauliZ(i)).
    device:
        PennyLane device name (e.g. 'default.qubit', 'default.qubit_gpu').
    shots:
        Number of shots for stochastic estimation. ``None`` uses the
        state‑vector device which returns exact expectation values.
    """

    def __init__(
        self,
        circuit,
        observables: Iterable[qml.operation.Operator],
        device: str = "default.qubit",
        shots: Optional[int] = None,
    ) -> None:
        self._observables = list(observables)
        dev = qml.device(device, wires=len(self._observables), shots=shots)
        self._qnode = qml.QNode(circuit, dev, interface="autograd")

    def evaluate(
        self,
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return expectation values for each parameter set.

        Parameters
        ----------
        parameter_sets:
            Sequence of parameter vectors. Each vector must match the circuit's
            number of parameters.
        """
        results: List[List[float]] = []
        for params in parameter_sets:
            # QNode returns a tuple of expectation values
            values = self._qnode(params)
            results.append([float(v) for v in values])
        return results


__all__ = ["FastBaseEstimator"]
