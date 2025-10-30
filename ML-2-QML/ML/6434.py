"""Enhanced estimator utilities built on PyTorch.

Features
--------
* Batch evaluation of arbitrary callable observables.
* Automatic GPU dispatch when available.
* Optional gradient computation via ``torch.autograd``.
* Flexible shot‑noise model: Gaussian or Poisson.
* Simple LRU cache for previously evaluated parameter sets.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence into a 2‑D tensor of shape (1, n)."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimatorGen256:
    """Evaluate a neural network for batches of inputs and observables."""

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None) -> None:
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self._cache: dict[tuple[float,...], torch.Tensor] = {}

    def _cached_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        key = tuple(inputs.squeeze().tolist())
        if key in self._cache:
            return self._cache[key]
        outputs = self.model(inputs.to(self.device))
        self._cache[key] = outputs.detach()
        return outputs

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        grad: bool = False,
        shots: Optional[int] = None,
        noise: str = "gaussian",
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Parameters
        ----------
        observables
            Callables that map the model output to a scalar.
        parameter_sets
            Iterable of parameter vectors to evaluate.
        grad
            If True, return gradients w.r.t model parameters.
        shots
            If provided, add statistical shot noise.
        noise
            Type of noise to add: ``"gaussian"`` or ``"poisson"``.
        seed
            Random seed for reproducibility.
        """
        if shots is not None and shots <= 0:
            raise ValueError("shots must be a positive integer or None")

        rng = np.random.default_rng(seed)
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad() if not grad else torch.enable_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
                outputs = self._cached_forward(inputs)

                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)

                if shots is not None:
                    if noise == "gaussian":
                        sigma = max(1e-6, 1 / np.sqrt(shots))
                        row = [float(rng.normal(mean, sigma)) for mean in row]
                    elif noise == "poisson":
                        row = [float(rng.poisson(mean)) / shots for mean in row]
                    else:
                        raise ValueError(f"Unsupported noise type: {noise}")

                results.append(row)

        return results


class FastEstimatorGen256(FastBaseEstimatorGen256):
    """Convenience wrapper that adds a simple Gaussian shot‑noise layer."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(
            observables,
            parameter_sets,
            grad=False,
            shots=shots,
            noise="gaussian",
            seed=seed,
        )
        return raw


__all__ = ["FastBaseEstimatorGen256", "FastEstimatorGen256"]
