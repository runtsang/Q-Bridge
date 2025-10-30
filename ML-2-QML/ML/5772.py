"""Enhanced lightweight estimator utilities implemented with PyTorch modules and optional preprocessing."""
from __future__ import annotations

from collections.abc import Iterable, Sequence, Callable
from typing import List, Optional, Any

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D tensor of shape (batch, 1) or (batch, n)."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """Evaluate neural networks with optional preprocessing and postprocessing."""

    def __init__(
        self,
        model: nn.Module,
        *,
        preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        postprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.model = model
        self.preprocess = preprocess
        self.postprocess = postprocess

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute observables for each parameter set."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                if self.preprocess is not None:
                    inputs = self.preprocess(inputs)
                outputs = self.model(inputs)
                if self.postprocess is not None:
                    outputs = self.postprocess(outputs)
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
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
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["FastBaseEstimator", "FastEstimator"]
