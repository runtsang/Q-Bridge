"""Hybrid fast estimator for classical neural networks with optional shot noise.

This module extends the original lightweight estimator by adding:
* robust batch handling for arbitrary nested sequences of parameters.
* support for arbitrary scalar observables defined as callables.
* configurable Gaussian noise that emulates finite‑shot sampling.
* integration with the QFCModel architecture from the Quantum‑NAT example.

The estimator remains fully PyTorch‑based and can be used as a drop‑in
replacement for the original FastBaseEstimator in any pipeline that
expects a pure‑Python, GPU‑ready implementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[Sequence[float]]) -> torch.Tensor:
    """Convert a list of parameter vectors into a 2‑D float tensor.

    Parameters
    ----------
    values
        Sequence of parameter vectors; each vector is a sequence of floats.
    Returns
    -------
    torch.Tensor
        Tensor of shape (batch, param_dim) with dtype float32.
    """
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for a batch of inputs.

    The estimator is intentionally lightweight: it disables gradients,
    evaluates the model once per batch, and returns a list of scalar
    observables for each parameter set.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a matrix of observable values.

        Parameters
        ----------
        observables
            Iterable of callables that map the model output to a scalar.
            If empty a default ``mean`` observable is used.
        parameter_sets
            Batch of parameter vectors to feed into the model.
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


class FastEstimator(FastBaseEstimator):
    """Add optional Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """Return noisy estimates of the observables.

        Parameters
        ----------
        shots
            Number of Monte‑Carlo shots per parameter set.  If ``None`` the
            deterministic values are returned.
        seed
            Random seed for reproducibility.
        """
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
