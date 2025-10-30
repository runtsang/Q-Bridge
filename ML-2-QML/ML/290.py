"""Hybrid classical estimator with batched evaluation, GPU support and optional gradient computation.

The FastBaseEstimator class accepts a PyTorch model and evaluates a set of scalar observables for
multiple batches of parameters. It automatically moves the model to a GPU if available, and can
return analytic gradients of the observables with respect to the model parameters.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.autograd import grad

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a 2‑D tensor suitable for batch evaluation."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for many parameter sets and observables."""

    def __init__(self, model: nn.Module, device: torch.device | str | None = None) -> None:
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return the mean of each observable for every parameter set."""
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

    def compute_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[torch.Tensor]]:
        """Return gradients of each observable w.r.t. model parameters for every parameter set."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        grads: List[List[torch.Tensor]] = []
        self.model.train()
        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device)
            outputs = self.model(inputs)
            grad_list: List[torch.Tensor] = []
            for observable in observables:
                value = observable(outputs)
                if isinstance(value, torch.Tensor):
                    val = value.mean()
                else:
                    val = torch.tensor(value, device=self.device, requires_grad=True)
                grad_vals = grad(val, self.model.parameters(), retain_graph=True, allow_unused=True)
                # Flatten gradients into a single vector per observable
                flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_vals if g is not None])
                grad_list.append(flat_grad.cpu())
            grads.append(grad_list)
        return grads


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


__all__ = ["FastBaseEstimator", "FastEstimator"]
