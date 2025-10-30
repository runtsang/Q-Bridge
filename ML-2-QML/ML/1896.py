"""
FastBaseEstimator – Classical (PyTorch) implementation with GPU support and
automatic differentiation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn, Tensor

ScalarObservable = Callable[[Tensor], Tensor | float]
ComplexObservable = Callable[[Tensor], Tensor | complex]


def _ensure_batch(values: Sequence[float]) -> Tensor:
    """Convert a list/tuple of floats into a 2‑D float32 tensor."""
    t = torch.as_tensor(values, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t


class FastBaseEstimator:
    """Evaluate a PyTorch neural network for many parameter sets and observables.

    The estimator is agnostic to the device – the model and its inputs will be moved
    to ``device`` automatically.  Observables can return scalars or complex tensors.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None) -> None:
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable | ComplexObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a list of observable values for each parameter set."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
                    if isinstance(value, Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results

    def jacobian(
        self,
        parameter_sets: Sequence[Sequence[float]],
        observables: Iterable[ScalarObservable | ComplexObservable],
    ) -> List[List[float]]:
        """Return the Jacobian of each observable w.r.t. the input parameters."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        jacobians: List[List[float]] = []

        self.model.train()
        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device).requires_grad_(True)
            outputs = self.model(inputs)
            row: List[float] = []
            for obs in observables:
                val = obs(outputs).sum()
                val.backward()
                grad = inputs.grad.detach().cpu().numpy().flatten()
                row.append(float(grad.mean()))
                inputs.grad.zero_()
            jacobians.append(row)
        return jacobians


class FastEstimator(FastBaseEstimator):
    """Wraps FastBaseEstimator and adds Gaussian shot noise to the outputs."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable | ComplexObservable],
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
