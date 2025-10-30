"""Enhanced estimator utilities built on PyTorch.

This module adds:
* batched parameter support via `_ensure_batch`
* optional shot‑noise simulation in `evaluate`
* gradient computation in `evaluate_with_gradients`
* device selection for GPU acceleration
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Optional
import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables with optional gradient support."""

    def __init__(self, model: nn.Module, device: torch.device | str | None = None) -> None:
        self.model = model.to(device or torch.device("cpu"))
        self.device = self.model.device

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        if parameter_sets is None:
            raise ValueError("parameter_sets must be provided")
        observables = list(observables or [lambda x: x.mean(dim=-1)])
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
                outputs = self._forward(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
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
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def evaluate_with_gradients(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> Tuple[List[List[float]], List[List[torch.Tensor]]]:
        """Return expectation values and gradients of each observable w.r.t. parameters."""
        if parameter_sets is None:
            raise ValueError("parameter_sets must be provided")
        observables = list(observables or [lambda x: x.mean(dim=-1)])
        results: List[List[float]] = []
        grads: List[List[torch.Tensor]] = []
        self.model.train()
        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device)
            inputs.requires_grad_(True)
            outputs = self._forward(inputs)
            row_vals: List[float] = []
            row_grads: List[torch.Tensor] = []
            for observable in observables:
                value = observable(outputs)
                if isinstance(value, torch.Tensor):
                    scalar = value.mean()
                else:
                    scalar = torch.tensor(value, dtype=torch.float32, device=self.device)
                scalar.backward(retain_graph=True)
                grad = inputs.grad.detach().clone()
                row_vals.append(float(scalar.cpu()))
                row_grads.append(grad.cpu())
                inputs.grad.zero_()
            results.append(row_vals)
            grads.append(row_grads)
        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy_results = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy_results.append(noisy_row)
            results = noisy_results
        return results, grads


__all__ = ["FastBaseEstimator"]
