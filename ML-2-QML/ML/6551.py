"""Hybrid estimator combining classical PyTorch model evaluation with optional convolutional preprocessing and Gaussian shot noise."""

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


class ConvFilter(nn.Module):
    """A simple 2‑D convolution filter that mimics the quantum quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply convolution to a single kernel‑sized patch."""
        x = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()


class HybridBaseEstimator:
    """
    Evaluate a PyTorch model with optional convolutional preprocessing and Gaussian shot noise.
    """
    def __init__(
        self,
        model: nn.Module,
        filter: nn.Module | None = None,
    ) -> None:
        self.model = model
        self.filter = filter

    def _prepare_input(self, params: Sequence[float]) -> torch.Tensor:
        tensor = _ensure_batch(params)
        if self.filter is not None:
            if tensor.ndim == 2 and tensor.shape[1] == self.filter.kernel_size ** 2:
                patch = tensor.reshape(-1, self.filter.kernel_size, self.filter.kernel_size)
                filtered = torch.stack([self.filter(p) for p in patch])
                return filtered
            return torch.tensor([float(self.filter(p)) for p in tensor], dtype=torch.float32)
        return tensor

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Compute observable values for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Functions that map a model output tensor to a scalar.
        parameter_sets : sequence of parameter sequences
            Each inner sequence is fed to the model after optional filtering.
        shots : int, optional
            If provided, Gaussian noise with variance 1/shots is added to each result.
        seed : int, optional
            Random seed for reproducibility when shots is not None.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = self._prepare_input(params)
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

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["HybridBaseEstimator"]
