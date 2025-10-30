"""Enhanced estimator utilities with vectorized evaluation, GPU support, and caching.

The class FastBaseEstimator now accepts a PyTorch model and evaluates it
in batch mode.  It supports optional device selection, caching of the
last evaluation, and a flexible observable interface.  FastEstimator
extends the base class by adding shot‑noise simulation with NumPy
random generators.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence into a batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Vectorised estimator for PyTorch models with optional GPU support.

    Parameters
    ----------
    model: nn.Module
        The neural network to evaluate.
    device: str | torch.device, optional
        Target device ('cpu' or 'cuda').  Defaults to 'cpu'.
    cache: bool, optional
        Enable caching of the last evaluated parameters to avoid
        redundant forward passes.
    """

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu", *, cache: bool = True) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.cache = cache
        self._last_params: torch.Tensor | None = None
        self._last_output: torch.Tensor | None = None

    def _forward(self, params: torch.Tensor) -> torch.Tensor:
        if self.cache and self._last_params is not None and torch.equal(params, self._last_params):
            return self._last_output
        self.model.eval()
        with torch.no_grad():
            out = self.model(params.to(self.device))
        if self.cache:
            self._last_params = params.clone().detach()
            self._last_output = out.clone().detach()
        return out

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate observables for each parameter set.

        Parameters
        ----------
        observables: Iterable[ScalarObservable]
            Callables that map the model output tensor to a scalar.
        parameter_sets: Sequence[Sequence[float]]
            Iterable of parameter vectors.  Each inner sequence is
            converted to a 1‑D tensor and evaluated in batch.

        Returns
        -------
        List[List[float]]
            A list of rows, each containing the value of every
            observable for a single parameter set.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        batch = torch.vstack([_ensure_batch(p) for p in parameter_sets]).to(self.device)
        outputs = self._forward(batch)
        for obs in observables:
            values = obs(outputs)
            if isinstance(values, torch.Tensor):
                values = values.cpu()
            results.append(values.tolist())
        # Transpose to match original API: rows correspond to parameter sets
        return [list(row) for row in zip(*results)]


class FastEstimator(FastBaseEstimator):
    """Estimator that injects Gaussian shot noise into deterministic results.

    Parameters
    ----------
    shots: int | None, optional
        Number of shots to use for noise simulation.  If None, no noise
        is added.
    seed: int | None, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
        *,
        cache: bool = True,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(model, device, cache=cache)
        self.shots = shots
        self.rng = np.random.default_rng(seed)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if self.shots is None:
            return raw
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(self.rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
