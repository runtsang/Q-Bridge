"""Enhanced hybrid estimator that can evaluate PyTorch models or PennyLane QNodes."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import pennylane as qml
from typing import Callable, Iterable, List, Sequence, Union

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
QuantumObservable = qml.operation.Operator

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence of floats into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastEstimator:
    """Hybrid estimator supporting PyTorch models and PennyLane QNodes.

    Parameters
    ----------
    model : nn.Module | qml.QNode
        The underlying model. For a PyTorch module the estimator runs
        the network on batches of parameters. For a PennyLane QNode
        the estimator evaluates the circuit for each parameter set.
    """
    def __init__(self, model: Union[nn.Module, qml.QNode]) -> None:
        self.model = model
        self._is_pennylane = isinstance(model, qml.QNode)

    def evaluate(
        self,
        observables: Iterable[Union[ScalarObservable, QuantumObservable]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate the model for each parameter set and observable.

        If the model is a PyTorch module, `observables` should be
        callables operating on the model's output tensor.  If the
        model is a PennyLane QNode, `observables` should be
        PennyLane operators and the QNode must accept an observable
        argument.

        Parameters
        ----------
        observables : Iterable
            Observables to evaluate.  For PyTorch modules these are
            functions; for PennyLane QNodes these are quantum
            operators.
        parameter_sets : Sequence[Sequence[float]]
            Sequences of parameter vectors.
        shots : int | None
            Number of shots to simulate.  If ``None`` the estimator
            returns exact expectation values.
        seed : int | None
            Random seed used when adding Gaussian shot noise.

        Returns
        -------
        List[List[float]]
            Results in row‑first order: each inner list contains the
            values for a single parameter set.
        """
        results: List[List[float]] = []

        if self._is_pennylane:
            # Evaluate PennyLane QNode
            for params in parameter_sets:
                row: List[float] = []
                for obs in observables:
                    if shots is not None:
                        dev = qml.device("default.qubit", wires=obs.wires, shots=shots)
                        qnode = qml.QNode(self.model.func, dev, interface="autograd")
                        val = qnode(params, obs)
                    else:
                        val = self.model(params, obs)
                    row.append(float(val))
                results.append(row)
        else:
            # Evaluate PyTorch model
            self.model.eval()
            with torch.no_grad():
                for params in parameter_sets:
                    inputs = _ensure_batch(params)
                    outputs = self.model(inputs)
                    row: List[float] = []
                    for obs in observables:
                        val = obs(outputs)
                        if isinstance(val, torch.Tensor):
                            scalar = float(val.mean().cpu())
                        else:
                            scalar = float(val)
                        row.append(scalar)
                    results.append(row)

        # Add shot noise if requested
        if shots is not None and seed is not None:
            rng = np.random.default_rng(seed)
            noisy_results: List[List[float]] = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy_results.append(noisy_row)
            return noisy_results

        return results

__all__ = ["FastEstimator"]
