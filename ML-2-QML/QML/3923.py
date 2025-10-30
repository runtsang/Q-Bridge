import pennylane as qml
import numpy as np
from collections.abc import Iterable, Sequence
from typing import List, Callable, Optional

class FastBaseEstimator:
    """Evaluate expectation values of a Pennylane QNode."""
    def __init__(self, circuit_fn: Callable, device: Optional[qml.Device] = None):
        self.circuit_fn = circuit_fn
        self.device = device or qml.device("default.qubit", wires=4)

    def evaluate(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
    ) -> List[List[complex]]:
        results: List[List[complex]] = []
        for params in parameter_sets:
            state = self.circuit_fn(*params)
            row = [qml.expval(o, state) for o in observables]
            results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Adds optional Poisson shot‑noise simulation to the quantum estimator."""
    def evaluate(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets, shots=shots)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [rng.poisson(abs(val) * shots) / shots for val in row]
            noisy.append(noisy_row)
        return noisy

class HybridModelQuantum:
    """
    Quantum variational model inspired by the Quantum‑NAT QFCModel.
    Uses Pennylane to encode a flattened image slice into qubits,
    applies a small random unitary circuit, and returns the statevector.
    """
    def __init__(self, wires: int = 4, seed: int = 42):
        self.wires = wires
        self.seed = seed
        self.device = qml.device("default.qubit", wires=wires, shots=1024)
        self.qnode = qml.QNode(self._qcircuit, self.device, interface="autograd")

    def _qcircuit(self, *params):
        for i, p in enumerate(params):
            qml.RY(p, wires=i)
        # apply a random unitary layer
        for _ in range(2):
            qml.RandomUnitary(self.wires, seed=self.seed)
        return qml.state()

    def __call__(self, *params):
        return self.qnode(*params)

__all__ = ["FastBaseEstimator", "FastEstimator", "HybridModelQuantum"]
