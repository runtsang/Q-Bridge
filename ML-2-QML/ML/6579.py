"""Hybrid fast estimator for classical neural networks with optional shot noise and convolutional feature extraction."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class ConvFilter(nn.Module):
    """Simple 2‑D convolutional filter emulating a quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Return mean activation after sigmoid threshold."""
        x = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

def Conv() -> ConvFilter:
    """Factory for the convolution filter used as a drop‑in observable."""
    return ConvFilter()

class FastBaseEstimator:
    """Evaluate a PyTorch model for batches of parameters and observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Return a matrix of observable values for each parameter set.

        Parameters
        ----------
        observables:
            Iterable of callables that map a model output tensor to a scalar.
        parameter_sets:
            Sequence of parameter vectors compatible with the model.
        shots, seed:
            If provided, inject Gaussian shot noise with variance 1/shots.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
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

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy

        return results

__all__ = ["FastBaseEstimator", "Conv"]
