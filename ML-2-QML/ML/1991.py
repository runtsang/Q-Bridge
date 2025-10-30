"""Hybrid estimator for neural networks with differentiable noise and gradient support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D tensor with a batch dimension."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class _LearnableNoise(nn.Module):
    """Add Gaussian noise with a trainable scale."""

    def __init__(self, init_scale: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.scale
        return x + noise


class FastBaseEstimator:
    """Evaluate a PyTorch model on batches of parameters with optional learnable noise
    and support for gradient computation."""
    def __init__(
        self,
        model: nn.Module,
        *,
        noise: bool = False,
        init_scale: float = 1.0,
    ) -> None:
        self.model = model
        self.noise_layer = _LearnableNoise(init_scale) if noise else None

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return deterministic or noisy predictions for each parameter set."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        self.model.eval()
        results: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                if self.noise_layer is not None:
                    outputs = self.noise_layer(outputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        return results

    def evaluate_with_grad(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return predictions and gradients w.r.t. inputs for each parameter set."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        self.model.train()
        results: List[List[float]] = []
        for params in parameter_sets:
            inputs = _ensure_batch(params).requires_grad_(True)
            outputs = self.model(inputs)
            if self.noise_layer is not None:
                outputs = self.noise_layer(outputs)
            grads: List[float] = []
            for obs in observables:
                val = obs(outputs)
                if isinstance(val, torch.Tensor):
                    val = val.mean()
                else:
                    val = torch.tensor(val, dtype=outputs.dtype, device=outputs.device)
                grad_val, = torch.autograd.grad(val, inputs, retain_graph=True)
                grads.append(float(grad_val.mean().cpu()))
            results.append(grads)
        return results


__all__ = ["FastBaseEstimator"]
