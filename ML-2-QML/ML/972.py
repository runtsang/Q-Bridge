"""Enhanced FastBaseEstimator with GPU support, batching, and gradient computation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[Sequence[float]]) -> torch.Tensor:
    """Convert a list of parameter vectors into a 2‑D torch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """GPU‑aware estimator for neural‑network models.

    The class accepts a ``torch.nn.Module`` and evaluates a list of
    observables over many parameter sets.  It supports optional
    batching and can be used as a lightweight forward‑pass API for
    quick prototyping or integration into larger pipelines.
    """

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        batch_size: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate all observables for every parameter set.

        Parameters
        ----------
        observables: iterable of callables
            Each callable receives the model output and returns a scalar
            or a tensor that can be reduced to a scalar.
        parameter_sets: sequence of parameter vectors
            Each vector is fed to the model as a single input.
        batch_size: int, optional
            If set, parameters are processed in chunks of this size.
        """
        if not observables:
            observables = [lambda out: out.mean(dim=-1)]

        results: List[List[float]] = []
        self.model.eval()

        param_tensor = _ensure_batch(parameter_sets).to(self.device)

        if batch_size is None or batch_size >= param_tensor.shape[0]:
            batches = [param_tensor]
        else:
            batches = torch.split(param_tensor, batch_size)

        with torch.no_grad():
            for batch in batches:
                outputs = self.model(batch)
                for params, output in zip(batch, outputs):
                    row: List[float] = []
                    for obs in observables:
                        val = obs(output)
                        if isinstance(val, torch.Tensor):
                            scalar = float(val.mean().cpu())
                        else:
                            scalar = float(val)
                        row.append(scalar)
                    results.append(row)
        return results

    def compute_gradients(
        self,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[torch.Tensor]:
        """
        Compute gradients of a scalar loss with respect to the model
        parameters for each parameter set.

        Parameters
        ----------
        loss_fn: callable
            Function that maps model output to a scalar loss.
        parameter_sets: sequence of parameter vectors
            Each vector is fed as a single input.
        """
        grads: List[torch.Tensor] = []
        self.model.train()

        param_tensor = _ensure_batch(parameter_sets).to(self.device)

        for params in param_tensor:
            params = params.unsqueeze(0)  # shape (1, n)
            params.requires_grad_(True)
            output = self.model(params)
            loss = loss_fn(output)
            loss.backward()
            grads.append(params.grad.clone().detach().cpu())
            self.model.zero_grad()
        return grads


class FastEstimator(FastBaseEstimator):
    """Adds optional shot‑noise simulation to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        noise_type: str = "gaussian",
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            if noise_type == "gaussian":
                noisy_row = [
                    rng.normal(mean, max(1e-6, 1 / shots)) for mean in row
                ]
            elif noise_type == "poisson":
                noisy_row = [
                    rng.poisson(max(1e-6, mean * shots)) / shots for mean in row
                ]
            else:
                raise ValueError(f"Unsupported noise type: {noise_type}")
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
