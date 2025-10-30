"""Combined classical estimator with optional self‑attention layer."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class ClassicalSelfAttention(nn.Module):
    """Self‑attention block that can be inserted into any torch model."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_lin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_lin   = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, rot: torch.Tensor, ent: torch.Tensor) -> torch.Tensor:
        # rot, ent shape (embed_dim, embed_dim)
        q = self.query_lin(x) @ rot
        k = self.key_lin(x)   @ ent
        scores = torch.softmax((q @ k.transpose(-1, -2)) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ x


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
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


__all__ = ["ClassicalSelfAttention", "FastBaseEstimator", "FastEstimator"]
