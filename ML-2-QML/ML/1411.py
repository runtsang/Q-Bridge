"""Enhanced fast estimator utilities built on PyTorch with optional batching and GPU support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Base class for evaluating neural‑network models on batched inputs.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to evaluate.
    device : torch.device | None, optional
        Device on which to perform inference. If ``None``, CUDA is used if available.
    """

    def __init__(self, model: nn.Module, device: torch.device | None = None) -> None:
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        device: torch.device | None = None,
    ) -> List[List[float]]:
        """Evaluate a set of observables on a batch of parameter vectors.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Callables that map a model output tensor to a scalar.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors.
        device : torch.device | None, optional
            Device on which to perform the computation. If ``None``, the instance's device is used.

        Returns
        -------
        List[List[float]]
            A list of rows; each row contains the scalar values for all observables
            evaluated on a single parameter set.
        """
        if device is None:
            device = self.device
        self.model.eval()
        results: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(device)
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
