"""Enhanced FastBaseEstimator with GPU support, batched inference, and gradient support.

This module extends the original lightweight estimator by adding:
* Automatic device selection (CUDA if available).
* Batched input handling for efficient inference.
* Optional Gaussian noise injection to mimic shot noise.
* Gradient evaluation via PyTorch autograd.
* Compatibility with scikit‑learn's BaseEstimator API.

The class can be used as a drop‑in replacement for the original, but offers richer functionality
for research and production workloads.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Callable, Iterable, List, Sequence, Optional
from collections.abc import Iterable as IterableABC

from sklearn.base import BaseEstimator as SklearnBaseEstimator, RegressorMixin

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator(SklearnBaseEstimator, RegressorMixin):
    """
    Evaluate a PyTorch model for a set of parameter vectors and scalar observables.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.
    device : str | torch.device | None, optional
        Target device. If ``None`` the class auto‑detects CUDA.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device | str] = None) -> None:
        super().__init__()
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

    def _prepare_batch(self, param_set: Sequence[Sequence[float]]) -> torch.Tensor:
        """Turn a list of parameter vectors into a batched tensor."""
        batch = torch.as_tensor(param_set, dtype=torch.float32, device=self.device)
        if batch.ndim == 1:
            batch = batch.unsqueeze(0)
        return batch

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Compute observables for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Functions that map a model output tensor to a scalar.
        parameter_sets : sequence of sequences
            Each inner sequence represents a parameter vector.

        Returns
        -------
        List[List[float]]
            A list of rows, one per parameter set, each containing the scalar values.
        """
        if not observables:
            observables = [lambda outputs: outputs.mean(dim=-1)]

        observables = list(observables)
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
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
        """
        Compute gradients of each observable w.r.t. model parameters for each parameter set.

        Returns
        -------
        List[List[torch.Tensor]]
            Nested list of gradients, one per parameter set and observable.
        """
        if not observables:
            observables = [lambda outputs: outputs.mean(dim=-1)]

        observables = list(observables)
        grads: List[List[torch.Tensor]] = []

        for params in parameter_sets:
            param_tensor = torch.as_tensor(params, dtype=torch.float32, device=self.device, requires_grad=True)
            outputs = self.model(param_tensor.unsqueeze(0))
            row_grads: List[torch.Tensor] = []
            for obs in observables:
                value = obs(outputs)
                if isinstance(value, torch.Tensor):
                    scalar = value.mean()
                else:
                    scalar = torch.tensor(value, dtype=torch.float32, device=self.device)
                grad = torch.autograd.grad(scalar, self.model.parameters(), retain_graph=True, allow_unused=True)
                grad_vector = torch.cat([g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
                                         for g, p in zip(grad, self.model.parameters())])
                row_grads.append(grad_vector.cpu())
            grads.append(row_grads)
        return grads

    def predict(self, X: Sequence[Sequence[float]]) -> List[List[float]]:
        """Alias for ``evaluate`` with a default observable that returns the model output mean."""
        return self.evaluate([lambda outputs: outputs.mean(dim=-1)], X)

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> "FastBaseEstimator":
        """No‑op fit method to satisfy the scikit‑learn API."""
        return self

    def add_noise(
        self,
        results: List[List[float]],
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Inject Gaussian noise to mimic shot noise.

        Parameters
        ----------
        results : List[List[float]]
            Raw deterministic results.
        shots : int | None
            Number of shots. If ``None`` no noise is added.
        seed : int | None
            Random seed for reproducibility.

        Returns
        -------
        List[List[float]]
            Noisy results.
        """
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


class FastEstimator(FastBaseEstimator):
    """
    Extends FastBaseEstimator with optional Gaussian shot noise.

    The ``evaluate`` method accepts ``shots`` and ``seed`` arguments to add noise.
    """

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        return self.add_noise(raw, shots=shots, seed=seed)


__all__ = ["FastBaseEstimator", "FastEstimator"]
