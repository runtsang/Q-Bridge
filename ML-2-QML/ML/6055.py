"""Hybrid classical estimator built on PyTorch.

The estimator evaluates an nn.Module over multiple parameter sets and a
collection of observables.  It supports optional Gaussian shot noise to
simulate measurement uncertainty, and it can handle hybrid models that
embed quantum layers via torchquantum.  The implementation is a thin
extension of the original FastBaseEstimator, adding noise handling
and a more flexible observable interface.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

Tensor = torch.Tensor
ScalarObservable = Callable[[Tensor], Tensor | float]


def _ensure_batch(values: Sequence[float]) -> Tensor:
    """Convert a list of values into a 2‑D batch tensor."""
    t = torch.as_tensor(values, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t


class HybridEstimator:
    """Evaluates a PyTorch model over a grid of parameters and observables.

    Parameters
    ----------
    model : nn.Module
        The model to be evaluated.  It may be a pure neural network or a
        hybrid network that contains quantum layers (e.g. a quanvolution
        filter implemented with torchquantum).

    Methods
    -------
    evaluate(observables, parameter_sets, shots=None, seed=None)
        Returns a list of rows, each row containing the value of every
        observable for a single parameter set.  When ``shots`` is
        provided, Gaussian noise with variance 1/shots is added to each
        result to mimic measurement shot noise.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        # Default observable: mean over last dimension
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["HybridEstimator"]
