"""Lightweight estimator utilities implemented with PyTorch modules, extended with batched evaluation, device support, and gradient computation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
LossFn = Callable[[torch.Tensor], torch.Tensor]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables with optional device support."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        device: str | torch.device | None = None,
    ) -> List[List[float]]:
        """
        Compute deterministic outputs for each parameter set.

        Parameters
        ----------
        observables
            Iterable of callables that map model output to a scalar.
        parameter_sets
            Sequence of parameter vectors.
        device
            Target device for computation. Defaults to the device of the model.

        Returns
        -------
        List[List[float]]
            Nested list of observable values.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        target_device = device or next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(target_device)
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

    def evaluate_with_grad(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        loss_fn: LossFn,
        device: str | torch.device | None = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute loss and gradients for each parameter set.

        Parameters
        ----------
        observables
            Iterable of callables that map model output to a scalar.
        parameter_sets
            Sequence of parameter vectors.
        loss_fn
            Callable that maps a scalar tensor to a loss tensor.
        device
            Target device for computation.

        Returns
        -------
        List[Tuple[Tensor, Tensor]]
            Each tuple contains (loss, gradient) for a parameter set.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        target_device = device or next(self.model.parameters()).device
        grads: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for params in parameter_sets:
            inputs = _ensure_batch(params).to(target_device)
            inputs.requires_grad_(True)
            outputs = self.model(inputs)
            row_loss = 0.0
            for observable in observables:
                val = observable(outputs)
                if isinstance(val, torch.Tensor):
                    val = val.mean()
                row_loss += loss_fn(val)
            row_loss.backward()
            grads.append((row_loss.detach().cpu(), inputs.grad.detach().cpu()))
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
        device: str | torch.device | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets, device=device)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
