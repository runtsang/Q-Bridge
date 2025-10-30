"""Enhanced estimator utilities based on PyTorch with gradient support and GPU acceleration."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D tensor for batch evaluation."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Fast, batched evaluator for PyTorch neural networks.

    The class mirrors the original ``FastBaseEstimator`` but adds optional
    device selection, GPU support, and a gradient helper.  The public
    ``evaluate`` method keeps the same signature as the seed so that
    downstream code continues to work unchanged.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None) -> None:
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a list of observable values for each parameter set.

        Parameters
        ----------
        observables:
            Callables that map the model output to a scalar (or Tensor).
            If omitted, the mean of the output is used.
        parameter_sets:
            Iterable of parameter vectors to evaluate.
        """
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

    def gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[List[float]]]:
        """Return gradients of each observable w.r.t. model parameters.

        The method evaluates each observable with autograd enabled and
        returns a nested list ``[parameter_set][observable][parameter]``.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        grads: List[List[List[float]]] = []

        self.model.train()
        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device)
            inputs.requires_grad_(True)
            outputs = self.model(inputs)
            row_grads: List[List[float]] = []
            for observable in observables:
                value = observable(outputs).mean()
                value.backward(retain_graph=True)
                grad = inputs.grad.squeeze().cpu().numpy().tolist()
                row_grads.append(grad)
                inputs.grad.zero_()
            grads.append(row_grads)
        return grads


class FastEstimator(FastBaseEstimator):
    """Estimator that optionally adds Gaussian shot noise to the predictions."""

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
