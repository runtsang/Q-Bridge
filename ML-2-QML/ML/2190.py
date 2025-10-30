"""Enhanced classical estimator with GPU support, batched observables, and analytic gradients."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn, Tensor

ScalarObservable = Callable[[Tensor], Tensor | float]


def _ensure_batch(values: Sequence[float]) -> Tensor:
    """Convert a sequence to a 2‑D float tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for many parameter sets and observables.

    Parameters
    ----------
    model : nn.Module
        Neural network to evaluate.  The module must accept a batch of
        parameter sets and return a tensor of shape ``(batch, output)``.
    device : torch.device | str | None
        Target device.  If ``None`` the model’s current device is used.
    """

    def __init__(self, model: nn.Module, device: torch.device | str | None = None) -> None:
        self.model = model.to(device or next(model.parameters()).device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return scalar observables for each parameter set."""
        observables = list(observables) or [lambda outputs: outputs.mean(-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

    def evaluate_with_gradient(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        grad_inputs: Sequence[int] | None = None,
    ) -> List[List[float]]:
        """Return observables and optional gradients w.r.t specified inputs.

        Parameters
        ----------
        grad_inputs
            Indices of parameters to differentiate with respect to.  If
            ``None`` gradients are omitted.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(-1)]
        results: List[List[float]] = []
        grads: List[List[float]] | None = None
        if grad_inputs is not None:
            grads = []

        self.model.train()
        for params in parameter_sets:
            inputs = _ensure_batch(params).requires_grad_(True)
            outputs = self.model(inputs)
            row: List[float] = []
            for obs in observables:
                val = obs(outputs)
                if isinstance(val, Tensor):
                    scalar = float(val.mean().cpu())
                else:
                    scalar = float(val)
                row.append(scalar)
            if grads is not None:
                grad_row: List[float] = []
                for idx in grad_inputs:
                    grad = torch.autograd.grad(
                        sum(row), inputs, retain_graph=True, allow_unused=True
                    )[0]
                    grad_row.append(float(grad[0, idx].item() if grad is not None else 0.0))
                grads.append(grad_row)
            results.append(row)

        if grads is not None:
            return [list(r) + list(g) for r, g in zip(results, grads)]
        return results


class FastEstimator(FastBaseEstimator):
    """Augments :class:`FastBaseEstimator` with shot‑noisy output.

    Parameters
    ----------
    shots : int | None
        Number of measurement shots.  If ``None`` no noise is added.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(model, device)
        self.shots = shots
        self.rng = np.random.default_rng(seed)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if self.shots is None:
            return raw
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [
                float(self.rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
