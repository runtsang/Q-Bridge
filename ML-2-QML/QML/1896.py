"""
FastBaseEstimator – Quantum (Pennylane) implementation with shot noise control and
analytic gradients.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from pennylane import QuantumDevice
from pennylane.operation import Operator

# --------------------------------------------------------------------------- #
# Helper: build a QNode from a parameterized circuit
# --------------------------------------------------------------------------- #
def _make_qnode(circuit, dev: QuantumDevice, observables: Iterable[Operator], shots: int | None = None):
    @qml.qnode(dev, interface="torch", diff_method="backprop", shots=shots)
    def qnode(params):
        circuit(params)
        return [qml.expval(op) for op in observables]

    return qnode


class FastBaseEstimator:
    """Evaluate a Pennylane circuit for many parameter sets and observables.

    The estimator accepts a circuit builder that receives a ``params`` tensor
    and applies gates to the device.  Observables are Pennylane Operators.
    """

    def __init__(
        self,
        circuit_builder: Callable[[pnp.ndarray], None],
        observables: Sequence[Operator],
        device_name: str = "default.qubit",
        wires: int | Sequence[int] = 1,
        shots: int | None = None,
    ) -> None:
        self.circuit_builder = circuit_builder
        self.observables = list(observables)
        self.dev = qml.device(device_name, wires=wires, shots=shots)
        self.qnode = _make_qnode(circuit_builder, self.dev, self.observables, shots=shots)

    def evaluate(
        self,
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a list of expectation values for each parameter set."""
        results: List[List[float]] = []
        for params in parameter_sets:
            # Pennylane expects a NumPy array; use pnp for autograd compatibility
            arr = pnp.array(params, dtype=pnp.float64)
            row = self.qnode(arr)
            results.append([float(val) for val in row])
        return results

    def gradients(
        self,
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return the gradients of each observable w.r.t. the parameters."""
        grads: List[List[float]] = []
        for params in parameter_sets:
            arr = pnp.array(params, dtype=pnp.float64)
            grad = self.qnode.grad(arr).reshape(-1).tolist()
            grads.append(grad)
        return grads


class FastEstimator(FastBaseEstimator):
    """Same as FastBaseEstimator but exposes a ``shots`` argument for noise control."""

    def __init__(
        self,
        circuit_builder: Callable[[pnp.ndarray], None],
        observables: Sequence[Operator],
        device_name: str = "default.qubit",
        wires: int | Sequence[int] = 1,
        shots: int | None = None,
    ) -> None:
        super().__init__(circuit_builder, observables, device_name, wires, shots=shots)


__all__ = ["FastBaseEstimator", "FastEstimator"]
