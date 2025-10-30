"""Hybrid estimator that wraps a PyTorch model and optionally adds shot noise.

The class retains the lightweight design of the original FastBaseEstimator but
introduces a training API and a stochastic evaluation mode that mimics
quantum shot noise.  This allows a classical pipeline to be swapped in for
the quantum part in a hybrid workflow.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Callable, Sequence

# Type alias for an observable function that returns a tensor or scalar
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Ensure inputs are 2‑D batch tensors."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Base class from the seed – unchanged."""
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


class HybridEstimator(FastBaseEstimator):
    """Hybrid estimator that adds optional Gaussian shot noise and a training API."""

    def __init__(
        self,
        model: nn.Module,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(model)
        self.shots = shots
        self.rng = np.random.default_rng(seed)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate the model and optionally add shot noise."""
        raw = super().evaluate(observables, parameter_sets)
        shots_to_use = shots if shots is not None else self.shots
        if shots_to_use is None:
            return raw
        rng = np.random.default_rng(seed if seed is not None else None)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots_to_use))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy

    def train(
        self,
        data_loader,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        epochs: int,
    ) -> None:
        """Simple training loop for the underlying PyTorch model."""
        self.model.train()
        for epoch in range(epochs):
            for batch in data_loader:
                inputs, targets = batch
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

__all__ = ["HybridEstimator"]
