"""Minimalistic estimator for PennyLane variational circuits with shot noise, gradients, and training support."""

import pennylane as qml
import pennylane.numpy as np
from typing import Iterable, List, Sequence, Optional

class FastEstimator:
    """
    A lightweight estimator for PennyLane QNodes that can evaluate expectation values,
    add shot noise, compute gradients, and run a simple training loop.
    """

    def __init__(self, qnode: qml.QNode):
        self.qnode = qnode
        self.num_params = qnode.num_params

    def evaluate(
        self,
        observables: Iterable[qml.operation.Operator | qml.measurements.MeasurementProcess],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        # Configure shots for the underlying device
        self.qnode.device.shots = shots if shots is not None else None
        if seed is not None:
            np.random.seed(seed)

        results: List[List[complex]] = []
        for params in parameter_sets:
            exp_vals = self.qnode(params, observables=observables)
            if np.isscalar(exp_vals):
                exp_vals = [exp_vals]
            results.append([val for val in exp_vals])
        return results

    def gradients(
        self,
        observables: Iterable[qml.operation.Operator | qml.measurements.MeasurementProcess],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[np.ndarray]]:
        grad_func = qml.grad(self.qnode, argnum=0)
        grads: List[List[np.ndarray]] = []

        for params in parameter_sets:
            grad_vals = grad_func(params)
            grads.append(list(grad_vals))
        return grads

    def train(
        self,
        observables: Iterable[qml.operation.Operator | qml.measurements.MeasurementProcess],
        train_data: Sequence[tuple[Sequence[float], Sequence[float]]],
        *,
        epochs: int = 10,
        learning_rate: float = 1e-3,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[float]:
        if seed is not None:
            np.random.seed(seed)

        params_list = np.array([p for p, _ in train_data], dtype=np.float64)
        targets = np.array([t for _, t in train_data], dtype=np.float64)

        loss_history: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for params, target in zip(params_list, targets):
                def loss_fn(pars):
                    preds = self.qnode(pars, observables=observables)
                    return np.sum((np.array(preds) - np.array(target)) ** 2)

                grad_fn = qml.grad(loss_fn)
                grads = grad_fn(params)
                params -= learning_rate * grads
                epoch_loss += loss_fn(params)
            loss_history.append(epoch_loss / len(train_data))
        return loss_history


__all__ = ["FastEstimator"]
