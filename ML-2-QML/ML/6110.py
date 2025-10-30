"""Enhanced FastBaseEstimator with GPU, autograd, and shot‑noise support."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn, Tensor

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
ScalarObservable = Callable[[Tensor], Tensor | float]


def _ensure_batch(values: Sequence[float]) -> Tensor:
    """Convert a 1‑D sequence into a batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


# --------------------------------------------------------------------------- #
# Core estimator classes
# --------------------------------------------------------------------------- #
class BaseEstimator:
    """Base class that evaluates a PyTorch model for multiple parameter sets."""
    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.model.eval()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        grad_mode: bool = False,
    ) -> List[List[float]]:
        """
        Compute observables for each parameter set.
        If grad_mode is True, gradients are tracked during the forward pass.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        if grad_mode:
            self.model.train()
            with torch.enable_grad():
                for params in parameter_sets:
                    inp = _ensure_batch(params).to(self.device)
                    out = self.model(inp)
                    row = []
                    for obs in observables:
                        val = obs(out)
                        if isinstance(val, Tensor):
                            val = val.mean()
                        row.append(float(val.cpu()))
                    results.append(row)
            self.model.eval()
        else:
            with torch.no_grad():
                for params in parameter_sets:
                    inp = _ensure_batch(params).to(self.device)
                    out = self.model(inp)
                    row = []
                    for obs in observables:
                        val = obs(out)
                        if isinstance(val, Tensor):
                            val = val.mean()
                        row.append(float(val.cpu()))
                    results.append(row)
        return results


class FastEstimator(BaseEstimator):
    """Adds shot‑noise simulation to BaseEstimator."""
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        grad_mode: bool = False,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets, grad_mode=grad_mode)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [rng.normal(loc=mean, scale=max(1e-6, 1 / shots)) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["BaseEstimator", "FastEstimator"]
