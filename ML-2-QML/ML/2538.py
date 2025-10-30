"""FastBaseEstimator extended with quanvolution support and noise injection."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Callable, Iterable, List, Sequence

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class QuanvolutionFilter(nn.Module):
    """A lightweight 2×2 patch convolution that emulates the quantum‑inspired filter."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)


class FastBaseEstimator:
    """Hybrid estimator that can wrap any nn.Module and optionally prepend a quanvolution filter."""
    def __init__(self, model: nn.Module, *, use_quanvolution: bool = False) -> None:
        if use_quanvolution:
            self.model = nn.Sequential(QuanvolutionFilter(), model)
        else:
            self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate the wrapped model for a list of parameter sets and observables.
        Parameters
        ----------
        observables:
            Callables mapping a model output tensor to a scalar.
            If None, defaults to the mean over the last dimension.
        parameter_sets:
            Iterable of parameter sequences to be fed as a single input tensor.
            If None, the model is evaluated once with its current parameters.
        shots:
            If provided, Gaussian noise with variance 1/shots is added to each result.
        seed:
            Random seed for reproducibility of noise.
        """
        if parameter_sets is None:
            parameter_sets = [()]
        if observables is None:
            observables = [lambda outputs: outputs.mean(dim=-1)]
        self.model.eval()
        results: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                if params:
                    inputs = _ensure_batch(params)
                else:
                    inputs = torch.zeros((1, 1, 28, 28))
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy = [[float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row] for row in results]
        return noisy


__all__ = ["FastBaseEstimator", "QuanvolutionFilter"]
