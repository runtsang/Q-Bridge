"""Hybrid fully‑connected layer – classical implementation.

The class mimics the quantum ``FCL`` API while using a lightweight
feed‑forward network.  It includes a ``FastEstimator`` helper that
provides deterministic evaluation and optional shot‑noise emulation,
inspired by the FastBaseEstimator pattern.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, Sequence, Callable, List
from collections.abc import Iterable as IterableABC

# --------------------------------------------------------------------------- #
# Helper: scalar observable
# --------------------------------------------------------------------------- #
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a parameter list to a batched tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


# --------------------------------------------------------------------------- #
# FastEstimator – deterministic with optional shot noise
# --------------------------------------------------------------------------- #
class FastEstimator:
    """Deterministic estimator that optionally adds Gaussian shot noise."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: IterableABC[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


# --------------------------------------------------------------------------- #
# HybridFCL – classical fully‑connected layer
# --------------------------------------------------------------------------- #
class HybridFCL(nn.Module):
    """Classical fully‑connected layer that emulates the quantum FCL API."""

    def __init__(self, n_features: int = 1, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Return the network output for a single parameter vector."""
        inputs = torch.as_tensor(list(thetas), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = self.net(inputs).squeeze(-1)
        return out.detach().cpu().numpy()

    def estimator(self) -> FastEstimator:
        """Return a helper that can evaluate batches with optional shot noise."""
        return FastEstimator(self)


__all__ = ["HybridFCL", "FastEstimator"]
