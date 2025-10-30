"""Enhanced estimator using PyTorch with optional Gaussian noise and softmax sampler.

The implementation expands upon the original FastBaseEstimator by
vectorising the evaluation of a PyTorch model, adding a convenient
SamplerQNN class, and exposing an optional shot‑noise wrapper.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn.functional import softmax
from typing import Callable, Iterable, List, Sequence

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[Sequence[float]]) -> torch.Tensor:
    """Convert a list of parameter vectors into a 2‑D batch tensor."""
    array = np.array(values, dtype=np.float32)
    return torch.as_tensor(array)


class FastBaseEstimator:
    """Evaluate a PyTorch model for multiple parameter sets and observables."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a matrix of observable values for each parameter set."""
        obs = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            batch = _ensure_batch(parameter_sets)
            outputs = self.model(batch)

            for i, _ in enumerate(parameter_sets):
                row: List[float] = []
                for func in obs:
                    value = func(outputs[i])
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Wraps FastBaseEstimator to add Gaussian shot noise."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        base = super().evaluate(observables, parameter_sets)
        if shots is None:
            return base

        rng = np.random.default_rng(seed)
        noisy = [
            [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            for row in base
        ]
        return noisy


class SamplerQNN(nn.Module):
    """A classical softmax sampler that mirrors the quantum SamplerQNN."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return softmax(self.net(inputs), dim=-1)


__all__ = ["FastBaseEstimator", "FastEstimator", "SamplerQNN"]
