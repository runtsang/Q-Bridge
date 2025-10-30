from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence of scalars to a 2‑D tensor of shape (1, N)."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """
    Lightweight estimator that evaluates a PyTorch model on batches of input
    parameters and returns scalar observables.
    """
    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Parameters
        ----------
        observables:
            Callables that map a model output tensor to a scalar.
            If empty, the mean over the last dimension is used.
        parameter_sets:
            Iterable of input parameter sequences (e.g. list of lists).
        Returns
        -------
        List of results, one per parameter set, each a list of observable values.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        self.model.eval()
        results: List[List[float]] = []
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
    """
    Extends FastBaseEstimator with optional Gaussian shot noise.
    """
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Parameters
        ----------
        shots:
            If provided, Gaussian noise with std = 1 / sqrt(shots) is added to each
            observable value to emulate finite sampling.
        seed:
            Random seed for reproducibility.
        """
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
