"""Hybrid estimator for classical PyTorch models with optional Gaussian shot noise.

This module defines ``HybridEstimator`` for evaluating neural networks on batches of
parameter sets and a set of scalar observables.  It also provides a lightweight
``FastEstimator`` subclass that adds Gaussian shot‑noise to the deterministic
results, mirroring the behaviour of the quantum counterpart.  The ``FCL`` helper
creates a fully‑connected layer that can be used as a drop‑in replacement in
experiments that originally used a quantum circuit.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of floats into a 2‑D torch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridEstimator:
    """Evaluate a PyTorch model over batches of parameters and observables."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a matrix of scalar observable values.

        Parameters
        ----------
        observables:
            Callables that map a model output tensor to a scalar or a 1‑D tensor.
        parameter_sets:
            Sequence of parameter vectors to feed into the model.

        Returns
        -------
        List[List[float]]:
            Each inner list contains the observable values for a single parameter set.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results


class FastEstimator(HybridEstimator):
    """Add optional Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


def FCL(n_features: int = 1) -> nn.Module:
    """Return a simple fully‑connected PyTorch layer mimicking a quantum FCL.

    The layer applies a linear transform followed by a tanh activation and
    returns the mean of the output.
    """
    class FullyConnectedLayer(nn.Module):
        def __init__(self, n_features: int = 1) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().numpy()

    return FullyConnectedLayer()


__all__ = ["HybridEstimator", "FastEstimator", "FCL"]
