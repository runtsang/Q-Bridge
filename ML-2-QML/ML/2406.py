"""
A hybrid classical sampler that extends the original SamplerQNN architecture.
It integrates the FastEstimator/ FastBaseEstimator pipeline for efficient
batch evaluation and optional shot‑noise simulation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections.abc import Iterable, Sequence
from typing import Callable, List

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a list of floats into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """
    Lightweight estimator that evaluates a PyTorch model for a batch of
    input parameter sets and a list of scalar observables.
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
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
        return results


class FastEstimator(FastBaseEstimator):
    """
    Adds optional Gaussian shot noise to the deterministic estimator.
    """
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
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


# --------------------------------------------------------------------------- #
# UnifiedSamplerQNN – classical neural sampler
# --------------------------------------------------------------------------- #
class UnifiedSamplerQNN(nn.Module):
    """
    A two‑layer MLP with optional dropout that mirrors the original
    SamplerQNN but supports configurable hidden size and noise injection.
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 4,
        output_dim: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a probability distribution over the output dimension."""
        return F.softmax(self.net(inputs), dim=-1)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate the network on a batch of parameters, optionally adding
        shot‑noise to the resulting expectation values.
        """
        estimator = FastEstimator(self)
        return estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)


__all__ = ["UnifiedSamplerQNN", "FastBaseEstimator", "FastEstimator"]
