"""Enhanced classical estimator with convolutional preprocessing and shot‑noise simulation.

The implementation extends the original FastBaseEstimator by:
* Optional 2‑D convolution filter (ConvFilter) applied to each input before
  model evaluation.
* Batch‑wise inference using PyTorch.
* Gaussian noise added to predictions to emulate finite‑shot effects.
* Support for multiple observable callables that transform model outputs.
"""

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
    """2‑D convolution filter that can be used as a preprocessing step."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the convolution and return the mean activation."""
        tensor = data.view(-1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3])


class FastBaseEstimator:
    """Classic estimator that evaluates a PyTorch model with optional filtering and noise."""

    def __init__(self, model: nn.Module, filter: Optional[nn.Module] = None) -> None:
        self.model = model
        self.filter = filter

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Return a list of observable values for each parameter set."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                if self.filter is not None:
                    # Assume the input corresponds to a 2‑D kernel
                    batch = inputs.view(-1, self.filter.kernel_size, self.filter.kernel_size)
                    filtered = self.filter(batch)
                    inputs = filtered.view(-1)
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

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy

        return results


__all__ = ["FastBaseEstimator", "ConvFilter"]
