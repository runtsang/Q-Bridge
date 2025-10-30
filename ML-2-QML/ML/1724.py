"""Enhanced classical estimator with autograd support and batched evaluation.

The API mirrors the original FastBaseEstimator but adds:
* ``evaluate_batch`` for efficient batched inputs.
* ``gradient`` that returns the gradient of each observable w.r.t a chosen
  parameter using PyTorch autograd.
* Optional GPU execution and a no‑gradient mode.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn, Tensor

ScalarObservable = Callable[[Tensor], Tensor | float]


def _ensure_batch(values: Sequence[float], device: torch.device | None = None) -> Tensor:
    """Convert a 1‑D sequence to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Base estimator for classical neural networks.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to evaluate.
    device : str | torch.device, optional
        Target device for computation (default: CPU).
    """

    def __init__(self, model: nn.Module, device: torch.device | str | None = None) -> None:
        self.model = model
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Deterministic evaluation of the model."""
        return self._eval_impl(observables, parameter_sets, use_grad=False)

    def evaluate_batch(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Tensor,
    ) -> Tensor:
        """Fast batched evaluation.  Returns a tensor of shape
        ``(batch, num_observables)``."""
        if parameter_sets.ndim == 1:
            parameter_sets = parameter_sets.unsqueeze(0)
        parameter_sets = parameter_sets.to(self.device)
        return self._eval_impl(observables, parameter_sets, use_grad=False, return_tensor=True)

    def gradient(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        param_idx: int,
    ) -> List[List[float]]:
        """Gradient of each observable w.r.t a selected parameter."""
        grads: List[List[float]] = []
        self.model.eval()
        for params in parameter_sets:
            inputs = _ensure_batch(params, device=self.device).requires_grad_(True)
            outputs = self.model(inputs)
            batch_grads = []
            for observable in observables:
                value = observable(outputs)
                if isinstance(value, Tensor):
                    grads_tensor = torch.autograd.grad(
                        outputs=value,
                        inputs=inputs,
                        grad_outputs=torch.ones_like(value),
                        retain_graph=True,
                        create_graph=False,
                    )[0]
                else:
                    # Scalar (float) observable
                    grads_tensor = torch.autograd.grad(
                        outputs=torch.tensor(value, device=self.device),
                        inputs=inputs,
                        retain_graph=True,
                        create_graph=False,
                    )[0]
                batch_grads.append(float(grads_tensor[0, param_idx].item()))
            grads.append(batch_grads)
        return grads

    def _eval_impl(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]] | Tensor,
        use_grad: bool,
        return_tensor: bool = False,
    ) -> List[List[float]] | Tensor:
        observables = list(observables) or [lambda x: x.mean(dim=-1)]
        self.model.eval()
        results: List[List[float]] | Tensor = []
        with torch.no_grad() if not use_grad else torch.enable_grad():
            if isinstance(parameter_sets, Tensor):
                batch_input = parameter_sets.to(self.device)
                outputs = self.model(batch_input)
                batch_results = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, Tensor):
                        batch_results.append(value.mean(dim=-1))
                    else:
                        batch_results.append(torch.tensor(value, device=self.device))
                # stack along last dimension
                stacked = torch.stack(batch_results, dim=-1)
                return stacked if return_tensor else stacked.cpu().tolist()
            else:
                for params in parameter_sets:
                    inputs = _ensure_batch(params, device=self.device)
                    outputs = self.model(inputs)
                    row: List[float] = []
                    for observable in observables:
                        value = observable(outputs)
                        if isinstance(value, Tensor):
                            scalar = float(value.mean().cpu())
                        else:
                            scalar = float(value)
                        row.append(scalar)
                    results.append(row)
                return results


class FastEstimator(FastBaseEstimator):
    """Adds shot‑noise simulation on top of FastBaseEstimator."""

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
