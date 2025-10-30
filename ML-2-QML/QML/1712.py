"""Quantum estimator built on PennyLane.

The class mirrors the classical FastBaseEstimator but operates on a
parametrised PennyLane QNode.  It supports vectorised evaluation, GPU
devices, and optional shot noise.  Observables can be any PennyLane
Operator or a NumPy matrix that is automatically wrapped as a
qml.Matrix.

"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.ops import Operator

ScalarObservable = Callable[[pnp.ndarray], pnp.ndarray | float]
ParameterSet = Sequence[float]
ParameterBatch = Sequence[ParameterSet]


class FastBaseEstimator:
    """Evaluate expectation values of quantum observables for a parametrised
    PennyLane circuit.

    Parameters
    ----------
    circuit:
        A ``qml.QNode`` or a ``qml.QCircuit`` (converted to a QNode inside).
    dev_name:
        Name of the PennyLane device to use (e.g. ``"default.qubit"``,
        ``"default.mixed"``, ``"default.qubit"``, or a custom GPU device).
    shots:
        Number of shots for measurement; ``None`` means exact statevector
        evaluation.

    Notes
    -----
    The class keeps the same public API as the classical version so that
    hybrid experiments can instantiate either implementation
    interchangeably.
    """

    def __init__(
        self,
        circuit: Union[qml.QNode, "qml.QCircuit"],
        dev_name: str = "default.qubit",
        shots: Optional[int] = None,
    ) -> None:
        self.dev_name = dev_name
        self.shots = shots
        self.device = qml.device(dev_name, wires=circuit.num_wires, shots=shots)
        if isinstance(circuit, qml.QNode):
            self._qnode = circuit
        else:
            self._qnode = qml.QNode(circuit, self.device)

    def _bind(self, parameter_values: ParameterSet) -> qml.QNode:
        """Return a new QNode with parameters bound to the supplied values."""
        return qml.bind(self._qnode, {k: v for k, v in zip(self._qnode.parameters, parameter_values)})

    def evaluate(
        self,
        observables: Iterable[Operator | np.ndarray],
        parameter_sets: ParameterBatch,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables:
            Iterable of ``Operator`` objects or NumPy matrices.  Arrays are
            automatically wrapped as ``qml.Matrix`` operators.
        parameter_sets:
            Iterable of parameter vectors.  Supports list of lists or a 2‑D
            NumPy array.

        Returns
        -------
        List[List[complex]]:
            Nested list where the outer list corresponds to parameter sets
            and the inner list to observables.
        """
        ops: List[Operator] = [
            op if isinstance(op, Operator) else qml.Matrix(op) for op in observables
        ]

        results: List[List[complex]] = []

        for params in parameter_sets:
            qnode = self._bind(params)
            row: List[complex] = []
            for op in ops:
                val = qnode(op)
                row.append(complex(val))
            results.append(row)

        return results


__all__ = ["FastBaseEstimator"]
