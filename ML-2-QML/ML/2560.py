"""Hybrid fully connected layer with fast estimator support.

This module defines a classical PyTorch implementation that mimics a quantum
fully‑connected layer.  It exposes an `evaluate` method inspired by
`FastBaseEstimator` and an optional `evaluate_noisy` that adds Gaussian shot
noise, mirroring the quantum side.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Sequence, Callable

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a 2‑D tensor for batch processing."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridFCL(nn.Module):
    """A classical fully connected layer that emulates a quantum circuit."""

    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        """Forward pass returning a tensor of activations."""
        values = _ensure_batch(thetas)
        return torch.tanh(self.linear(values))

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Return the mean activation as a NumPy array."""
        output = self.forward(thetas)
        expectation = output.mean(dim=0)
        return expectation.detach().cpu().numpy()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Fast deterministic estimator for multiple parameter sets.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output and returns a scalar or
            a tensor.  If empty, a default mean is used.
        parameter_sets : sequence of parameter sequences
            Each inner sequence contains the parameters for one evaluation.

        Returns
        -------
        results : list of lists
            Outer list over parameter sets, inner list over observables.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.forward(inputs)
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

    def evaluate_noisy(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Add Gaussian shot noise to the deterministic estimator.

        Parameters
        ----------
        shots : int, optional
            Number of shots; if None, returns deterministic results.
        seed : int, optional
            Random seed for reproducibility.
        """
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


def FCL() -> type:
    """Return the HybridFCL class for external usage."""
    return HybridFCL
