import pennylane as qml
from pennylane import numpy as npy
from collections.abc import Iterable, Sequence
from typing import List

class FastBaseEstimatorGen212:
    """Expectation‑value estimator for Pennylane QNodes with optional shot noise."""
    def __init__(self, qnode: qml.QNode, *, device: str = "default.qubit", shots: int | None = None):
        self.qnode = qnode
        self.device_name = device
        self.shots = shots

    def evaluate(self, observables: Iterable[qml.operation.Operator], parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        observables = list(observables)
        results: List[List[float]] = [[] for _ in parameter_sets]
        if self.shots is None:
            # Deterministic evaluation via state‑vector
            for i, params in enumerate(parameter_sets):
                state = self.qnode(*params)  # statevector
                row = []
                for obs in observables:
                    val = qml.expval(obs)(state)
                    row.append(float(val))
                results[i] = row
        else:
            # Shot‑noise simulation using a device with the requested shots
            dev = qml.device(self.device_name, wires=self.qnode.wires, shots=self.shots)
            def _circuit(*params):
                self.qnode(*params)
                return [qml.expval(obs) for obs in observables]
            qnode_shots = qml.QNode(_circuit, dev)
            for i, params in enumerate(parameter_sets):
                row = qnode_shots(*params)
                results[i] = [float(v) for v in row]
        return results

    def gradient(self, observables: Iterable[qml.operation.Operator], parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        """Return concatenated gradients of all observables with respect to all parameters."""
        observables = list(observables)
        grads: List[List[float]] = [[] for _ in parameter_sets]
        for i, params in enumerate(parameter_sets):
            grad_row: List[float] = []
            for obs in observables:
                grad_fn = qml.grad(lambda *p: qml.expval(obs)(self.qnode(*p)))
                grad = grad_fn(*params)
                grad_row.extend(grad.tolist())
            grads[i] = grad_row
        return grads

__all__ = ["FastBaseEstimatorGen212"]
