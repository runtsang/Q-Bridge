"""Hybrid fast estimator for torch models with optional self‑attention and shot‑noise.

This module defines FastBaseEstimator and FastEstimator.
FastBaseEstimator can wrap any torch.nn.Module or a factory that returns one.
An optional self‑attention block can be injected via a factory.
FastEstimator adds Gaussian noise to emulate shot noise.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class ClassicalSelfAttention(nn.Module):
    """PyTorch implementation of the self‑attention block used in the seed."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.softmax(q @ k.transpose(-1, -2) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v


class FastBaseEstimator:
    """Evaluate a torch model for batches of inputs and observables.

    Parameters
    ----------
    model : nn.Module | Callable[[int], nn.Module]
        The neural network to evaluate. If a callable is provided it must
        accept an integer (the embedding dimension) and return a nn.Module.
    self_attention_factory : Optional[Callable[[int], nn.Module]]
        Factory that returns a self‑attention module. If None, no attention
        is applied to the inputs.
    embed_dim : int
        Dimension of the embedding used by the attention block.
    """

    def __init__(
        self,
        model: nn.Module | Callable[[int], nn.Module],
        *,
        self_attention_factory: Optional[Callable[[int], nn.Module]] = None,
        embed_dim: int = 4,
    ) -> None:
        if isinstance(model, nn.Module):
            self.model = model
        else:
            self.model = model(embed_dim)
        if self_attention_factory is not None:
            self.self_attention = self_attention_factory(embed_dim)
        else:
            self.self_attention = None

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
                if self.self_attention is not None:
                    inputs = self.self_attention(inputs)
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


__all__ = ["FastBaseEstimator", "FastEstimator", "ClassicalSelfAttention"]
