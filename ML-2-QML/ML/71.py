"""Fast, GPU‑friendly estimator utilities built on PyTorch."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(
    values: Sequence[Sequence[float]] | Sequence[float]
) -> torch.Tensor:
    """Convert a list of parameter sets or a single set into a 2‑D tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class _BaseEstimatorBase:
    """Base class that manages device selection and common utilities."""

    def __init__(self, device: str | None = None):
        if device is None:
            # Prefer GPU if available
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._device = torch.device(device)

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self._device)


class FastBaseEstimator(_BaseEstimatorBase):
    """Evaluate neural networks for batches of inputs and observables."""

    def __init__(self, model: nn.Module, device: str | None = None) -> None:
        super().__init__(device)
        self.model = model.to(self._device)
        self.model.eval()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Compute expectations for a batch of parameter sets.

        Parameters
        ----------
        observables:
            Callables that map a model output tensor to a scalar value.
        parameter_sets:
            Iterable of parameter vectors to evaluate.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        with torch.no_grad():
            inputs = _ensure_batch(parameter_sets).to(self._device)
            outputs = self.model(inputs)

            for output in outputs:
                row: List[float] = []
                for obs in observables:
                    value = obs(output)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)

        return results


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""

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
        std = 1 / np.sqrt(shots)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, std))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
