"""Hybrid estimator combining PyTorch and optional feature extraction."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class HybridEstimator:
    """Evaluate a PyTorch model with optional feature extraction and shot‑noise simulation."""

    def __init__(
        self,
        model: nn.Module,
        *,
        feature_extractor: nn.Module | None = None,
    ) -> None:
        self.model = model
        self.feature_extractor = feature_extractor

    def _prepare_inputs(self, params: Sequence[float]) -> torch.Tensor:
        """Turn a parameter set into a batched tensor."""
        batch = torch.as_tensor(params, dtype=torch.float32)
        if batch.ndim == 1:
            batch = batch.unsqueeze(0)
        return batch

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute observables for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Functions that map a model output tensor to a scalar.
        parameter_sets : sequence of parameter sequences
            Each inner sequence is fed to the model.
        shots : int, optional
            If provided, Gaussian noise with variance 1/shots is added to each result.
        seed : int, optional
            Random seed for reproducibility of the shot noise.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = self._prepare_inputs(params)
                if self.feature_extractor is not None:
                    inputs = self.feature_extractor(inputs)
                outputs = self.model(inputs)

                row: List[float] = []
                for obs_fn in observables:
                    value = obs_fn(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noise_std = max(1e-6, 1.0 / shots)
            noisy_results = [
                [float(rng.normal(mean, noise_std)) for mean in row]
                for row in results
            ]
            return noisy_results
        return results


__all__ = ["HybridEstimator"]
