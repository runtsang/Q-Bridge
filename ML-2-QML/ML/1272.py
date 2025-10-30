"""Lightweight, batched estimator utilities built on PyTorch with optional GPU support and richer observables."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Base class for evaluating neural‑network models on batches of parameter sets.

    The implementation is intentionally lightweight: it does not depend on
    **torch.autograd** for inference, but it provides a convenient
    **torch.no_grad** context for deterministic evaluation.
    """

    def __init__(self, model: nn.Module, device: Optional[str] = None) -> None:
        self.model = model
        self.device = device or "cpu"
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Return a 2‑D NumPy array with shape (n_params, n_obs).

        Parameters
        ----------
        observables : Iterable[Callable[[torch.Tensor], torch.Tensor | float]]
            A list‑like of scalar‑valuing functions applied to the model
            outputs.  If no observables are provided, a default observable
            that returns the mean of all outputs is used.
        parameter_sets : Sequence[Sequence[float]]
            A sequence of parameter vectors; each inner sequence is
            interpreted as a single batch entry.
        """
        if not observables:
            def default_observable(outputs: torch.Tensor) -> torch.Tensor:
                return torch.mean(outputs, dim=-1)
            observables = [default_observable]

        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
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
        return np.array(results, dtype=np.float32)


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = raw + rng.normal(
            loc=0.0, scale=np.sqrt(1 / shots), size=raw.shape
        )
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
