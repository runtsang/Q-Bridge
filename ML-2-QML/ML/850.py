"""Enhanced FastBaseEstimator for PyTorch models with batched evaluation and optional noise.

The class accepts a torch.nn.Module and evaluates it on batches of parameter sets.
Observables are callables that receive the model output and return a scalar.
Optional Gaussian shot noise can be added to emulate measurement uncertainty.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D parameter sequence to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for many parameter sets and observables.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.
    device : str | torch.device, optional
        The device on which to run the model. ``'cpu'`` by default.
    """

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a list of rows, each containing the observable values for one
        parameter set.

        Parameters
        ----------
        observables : iterable
            Callables that map a model output to a scalar.
        parameter_sets : sequence of sequences
            List of parameter vectors to evaluate.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        # Prepare batch of parameters
        batch = torch.stack([_ensure_batch(params) for params in parameter_sets], dim=0)
        batch = batch.to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch)
            for out in outputs:
                row: List[float] = []
                for observable in observables:
                    value = observable(out)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Add optional Gaussian shot noise to the deterministic estimator.

    Parameters
    ----------
    shots : int | None, optional
        If provided, Gaussian noise with variance 1/shots is added to each mean.
    seed : int | None, optional
        Random seed for reproducibility.
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
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
