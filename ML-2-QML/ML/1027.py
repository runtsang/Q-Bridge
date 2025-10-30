"""Advanced Fast Estimator for PyTorch models with GPU support and automatic gradients."""

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


class AdvancedFastEstimator:
    """GPU‑accelerated estimator for neural networks with support for shot noise and gradients."""
    def __init__(self, model: nn.Module, device: str = 'cpu') -> None:
        self.model = model.to(device)
        self.device = torch.device(device)

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
                inputs = _ensure_batch(params).to(self.device)
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

    def evaluate_with_shots(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        raw = self.evaluate(observables, parameter_sets)
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def evaluate_batch(
        self,
        observables: Iterable[ScalarObservable],
        parameters: torch.Tensor,
    ) -> torch.Tensor:
        """Return a tensor of shape (batch, observables) with expectation values."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        parameters = parameters.to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(parameters)
            batch_results = []
            for observable in observables:
                batch_results.append(observable(outputs))
            return torch.stack(batch_results, dim=-1)

    def compute_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[torch.Tensor]]:
        """Return gradients of each observable w.r.t. each parameter."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        grads: List[List[torch.Tensor]] = []
        self.model.train()
        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device)
            inputs.requires_grad_(True)
            outputs = self.model(inputs)
            row_grads: List[torch.Tensor] = []
            for observable in observables:
                value = observable(outputs)
                if isinstance(value, torch.Tensor):
                    value = value.mean()
                else:
                    value = torch.tensor(value, device=self.device, requires_grad=True)
                value.backward()
                row_grads.append(inputs.grad.detach().clone())
                inputs.grad.zero_()
            grads.append(row_grads)
        return grads


__all__ = ["AdvancedFastEstimator"]
