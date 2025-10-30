"""Hybrid fast estimator with optional preprocessing and shot‑noise simulation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
Preprocess = Callable[[torch.Tensor], torch.Tensor]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a 2‑D torch Tensor (batch of size 1)."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a neural network for batches of parameters and observables.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to evaluate.
    preprocess : Optional[Preprocess], default None
        A callable applied to the raw input parameters before feeding them
        to ``model``.  This can be a quanvolution filter or any custom
        feature extractor.
    """

    def __init__(self, model: nn.Module, preprocess: Preprocess | None = None) -> None:
        self.model = model
        self.preprocess = preprocess

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a list of observable values for each parameter set.

        Parameters
        ----------
        observables
            Iterable of callables that accept a ``torch.Tensor`` (model
            outputs) and return a scalar or a tensor that can be reduced
            to a scalar.
        parameter_sets
            Sequence of parameter vectors.  Each vector is a sequence of
            floats of the same length.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                if self.preprocess is not None:
                    inputs = self.preprocess(inputs)
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
    """Same as :class:`FastBaseEstimator` but adds optional Gaussian shot noise.

    Parameters
    ----------
    shots : Optional[int], default None
        Number of shots to simulate.  When ``None`` the estimator is deterministic.
    seed : Optional[int], default None
        Seed for the random number generator used in shot noise simulation.
    """

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


__all__ = ["FastBaseEstimator", "FastEstimator"]
