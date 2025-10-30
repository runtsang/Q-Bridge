"""Hybrid classical estimator that extends FastBaseEstimator with advanced batching and noise handling.

This module merges the lightweight PyTorch evaluation logic from the original FastBaseEstimator
with a flexible interface that can accept arbitrary observables and add Gaussian shot noise.
The class is designed to be compatible with the original anchor path while providing a
scalable foundation for hybrid quantum‑classical workflows.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """
    Convert a 1‑D sequence of parameters into a 2‑D batch tensor.
    """
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridBaseEstimator:
    """
    Evaluate a PyTorch model for batches of parameters and arbitrary scalar observables.

    Parameters
    ----------
    model : nn.Module
        Any differentiable neural network that accepts a batch of input parameters.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
    ) -> List[List[float]]:
        """
        Compute scalar observables for each parameter set.

        Parameters
        ----------
        observables
            Callable functions that map model outputs to scalar values.
            If ``None`` a single observable returning the mean of the output
            is used.
        parameter_sets
            2‑D sequence of parameters to evaluate.  Each inner sequence
            should match the input shape expected by ``model``.

        Returns
        -------
        List[List[float]]
            A list of rows, one per parameter set, each row containing the
            observable values.
        """
        if parameter_sets is None:
            raise ValueError("parameter_sets must be provided")
        observables = list(observables or [lambda out: out.mean(dim=-1)])

        results: List[List[float]] = []
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
        return results

    def evaluate_noisy(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Same as :meth:`evaluate` but adds Gaussian shot noise to each observable.

        Parameters
        ----------
        shots
            Number of shots to use for noise sampling.  If ``None`` the method
            simply forwards to :meth:`evaluate`.
        seed
            Random seed for reproducibility.
        """
        raw = self.evaluate(observables, parameter_sets)
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


class HybridEstimator(HybridBaseEstimator):
    """
    Convenience subclass that exposes the noisy evaluation as ``evaluate``.
    """

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        return self.evaluate_noisy(observables, parameter_sets, shots=shots, seed=seed)


__all__ = ["HybridBaseEstimator", "HybridEstimator"]
