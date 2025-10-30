"""Enhanced estimator utilities with device-aware inference and gradient support."""

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


class FastBaseEstimator:
    """Base class for evaluating a torch.nn.Module over batches of inputs."""

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None) -> None:
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

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


class FastEstimator(FastBaseEstimator):
    """Fast estimator with optional shot noise and gradient evaluation."""

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

    def evaluate_with_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        create_graph: bool = False,
    ) -> List[List[np.ndarray]]:
        """Return gradients of each observable w.r.t. the input parameters.

        Parameters
        ----------
        observables
            Callables that map a model output tensor to a scalar (or tensor which is
            reduced to a scalar by ``mean``).
        parameter_sets
            Sequence of input parameter vectors.
        create_graph
            If ``True`` the returned gradients will be tensors that require gradients
            themselves, allowing higher‑order differentiation.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        grads: List[List[np.ndarray]] = []
        self.model.eval()
        for params in parameter_sets:
            param_tensor = _ensure_batch(params).requires_grad_(True).to(self.device)
            outputs = self.model(param_tensor)
            row_grads: List[np.ndarray] = []
            for observable in observables:
                value = observable(outputs)
                if isinstance(value, torch.Tensor):
                    scalar = value.mean()
                else:
                    scalar = torch.tensor(value, dtype=torch.float32, device=self.device)
                grad = torch.autograd.grad(
                    scalar, param_tensor, retain_graph=True, create_graph=create_graph
                )[0]
                row_grads.append(grad.detach().cpu().numpy().flatten())
            grads.append(row_grads)
        return grads


__all__ = ["FastBaseEstimator", "FastEstimator"]
