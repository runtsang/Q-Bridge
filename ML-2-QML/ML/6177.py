"""Enhanced estimator utilities built on PyTorch.

The class now:
* Handles batched input and optional GPU execution.
* Supports a list of callable observables, each receiving the model output.
* Caches results for identical parameter sets to avoid redundant forward passes.
* Provides a simple interface for early‑stopping style noise injection.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """
    Evaluate a PyTorch model for multiple parameter sets and observables.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.
    device : str | torch.device | None
        Execution device.  Defaults to CPU.
    cache_size : int
        Maximum number of unique parameter sets to cache.  ``0`` disables caching.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device] | None = None,
        cache_size: int = 0,
    ) -> None:
        self.model = model
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)
        self.cache_size = cache_size
        self._cache: Dict[Tuple[float,...], List[float]] = {}

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | torch.Tensor | None = None,
        *,
        batch_size: int = 64,
    ) -> List[List[float]]:
        """
        Compute observables for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output and returns a scalar
            or a tensor that is converted to a float.
        parameter_sets : sequence of sequences or torch.Tensor
            Each inner sequence contains the input parameters for the model.
        batch_size : int
            Number of parameter sets processed per forward pass.

        Returns
        -------
        List[List[float]]
            Outer list over parameter sets, inner list over observables.
        """
        if observables is None:
            observables = [lambda out: out.mean(dim=-1)]
        observables = list(observables)

        if parameter_sets is None:
            raise ValueError("parameter_sets must be provided")

        if isinstance(parameter_sets, torch.Tensor):
            param_tensor = parameter_sets.to(self.device)
        else:
            param_tensor = torch.as_tensor(
                parameter_sets, dtype=torch.float32, device=self.device
            )

        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for start in range(0, param_tensor.shape[0], batch_size):
                batch = param_tensor[start : start + batch_size]
                outputs = self.model(batch)

                for params, out in zip(
                    param_tensor[start : start + batch_size], outputs
                ):
                    key = tuple(params.cpu().numpy())
                    if self.cache_size and key in self._cache:
                        row = self._cache[key]
                    else:
                        row: List[float] = []
                        for obs in observables:
                            val = obs(out)
                            if isinstance(val, torch.Tensor):
                                scalar = float(val.mean().cpu())
                            else:
                                scalar = float(val)
                            row.append(scalar)
                        if self.cache_size:
                            if len(self._cache) >= self.cache_size:
                                # simple LRU eviction
                                self._cache.pop(next(iter(self._cache)))
                            self._cache[key] = row
                    results.append(row)

        return results


class FastEstimator(FastBaseEstimator):
    """
    Adds optional Gaussian shot noise to the deterministic estimator.

    Parameters
    ----------
    shots : int | None
        Number of shots to emulate.  ``None`` disables noise.
    seed : int | None
        Random seed for reproducibility.
    """

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | torch.Tensor | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
        batch_size: int = 64,
    ) -> List[List[float]]:
        raw = super().evaluate(
            observables, parameter_sets, batch_size=batch_size
        )
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
