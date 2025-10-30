"""Hybrid classical convolutional filter with FastEstimator API.

Combines learnable Conv2d with a parameterizable estimator interface.
Supports dynamic weight assignment and optional Gaussian shot noise.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class ConvGen219(nn.Module):
    """Classical convolutional filter with FastEstimator interface.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolution kernel.
    threshold : float, default 0.0
        Sigmoid threshold.
    bias : bool, default True
        Whether the convolution layer has a bias.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=bias)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations

    def run(self, data) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        activations = self.forward(tensor)
        return activations.mean().item()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate the model for a batch of parameter sets.

        Parameters are interpreted as new convolution kernel weights.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                weight_shape = self.conv.weight.shape
                weight_count = weight_shape.numel()
                if len(params)!= weight_count:
                    raise ValueError("Parameter count mismatch for ConvGen219 kernel.")
                new_weight = torch.as_tensor(params, dtype=torch.float32).view(weight_shape)
                # Assign new weights
                self.conv.weight.data.copy_(new_weight)
                # Run on dummy input to trigger forward
                outputs = self.forward(torch.zeros(1, 1, self.kernel_size, self.kernel_size))
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    scalar = float(val.mean().cpu()) if isinstance(val, torch.Tensor) else float(val)
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


__all__ = ["ConvGen219"]
