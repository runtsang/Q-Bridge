"""Enhanced FastBaseEstimator with batched inference and optional GPU support."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Callable, Iterable, List, Sequence, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _batchify(
    input_values: Sequence[Sequence[float]],
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Convert a list of parameter sets into a 2‑D torch.Tensor.
    The result is moved to *device* if provided.
    """
    tensor = torch.as_tensor(input_values, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model on batches of parameters."""

    def __init__(self, model: nn.Module, device: Optional[str] = None) -> None:
        self.model = model
        if device is not None:
            self.model.to(device)
        self.device = device

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Compute deterministic predictions for *observables* on the given
        *parameter_sets*.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        self.model.eval()
        inputs = _batchify(parameter_sets, device=self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
            results: List[List[float]] = []
            for out in outputs:
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds stochastic shot‑noise to the deterministic FastBaseEstimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Return noisy predictions. If *shots* is ``None`` the call falls back
        to the deterministic implementation.
        """
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
