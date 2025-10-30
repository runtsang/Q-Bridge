"""Enhanced estimator utilities using PyTorch with batched evaluation and optional shot noise."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float | np.ndarray]


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables with GPU support."""

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.model.eval()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]] | torch.Tensor,
        *,
        batch_size: int | None = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Compute observables for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output (torch.Tensor) and returns a scalar,
            a vector, or a NumPy array. Scalars are flattened to Python floats.
        parameter_sets : sequence of sequences or torch.Tensor
            Parameters to evaluate. If a torch.Tensor, it must be 2‑D with shape (N, P).
        batch_size : int | None
            Number of samples processed per forward pass. If None, all parameters are
            processed in a single batch.
        shots : int | None
            If set, Gaussian shot noise with variance 1/shots is added to each result.
        seed : int | None
            Random seed for shot noise reproducibility.
        """
        if isinstance(parameter_sets, torch.Tensor):
            params = parameter_sets.to(self.device)
        else:
            params = torch.as_tensor(parameter_sets, dtype=torch.float32, device=self.device)

        if params.ndim == 1:
            params = params.unsqueeze(0)

        if batch_size is None or batch_size <= 0:
            batch_size = params.shape[0]

        rng = np.random.default_rng(seed)
        results: List[List[float]] = []

        for start in range(0, params.shape[0], batch_size):
            batch = params[start : start + batch_size]
            with torch.no_grad():
                outputs = self.model(batch)

            for param_idx in range(batch.shape[0]):
                row: List[float] = []
                output = outputs[param_idx]
                for obs in observables:
                    val = obs(output)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    elif isinstance(val, np.ndarray):
                        scalar = float(val.mean())
                    else:
                        scalar = float(val)
                    if shots is not None:
                        std = max(1e-6, 1.0 / np.sqrt(shots))
                        scalar += rng.normal(0.0, std)
                    row.append(scalar)
                results.append(row)

        return results


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]] | torch.Tensor,
        *,
        batch_size: int | None = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        return super().evaluate(
            observables,
            parameter_sets,
            batch_size=batch_size,
            shots=shots,
            seed=seed,
        )


__all__ = ["FastBaseEstimator", "FastEstimator"]
