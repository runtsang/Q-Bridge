"""Hybrid classical convolutional filter with FastEstimator support.

This module combines the classical Conv filter from the original seed with
the lightweight FastBaseEstimator utilities.  The Conv class is a
torch.nn.Module that emulates a quanvolution layer, while the
FastEstimator class adds Gaussian shot noise to deterministic outputs,
mirroring the behaviour of the QML FastEstimator.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Sequence, Callable

# ----- Conv class -----
class Conv(nn.Module):
    """A classical 2‑D convolutional filter that mimics a quanvolution.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square filter.
    threshold : float, default 0.0
        Activation threshold applied after convolution.
    bias : bool, default True
        Whether to learn a bias term.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 bias: bool = True) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size,
                              bias=bias, stride=1, padding=0)
        # Initialise weights to a uniform range similar to a random quantum circuit
        nn.init.uniform_(self.conv.weight, -np.pi, np.pi)
        if bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Forward pass returning a scalar activation."""
        tensor = torch.as_tensor(data, dtype=torch.float32)
        if tensor.ndim!= 2:
            raise ValueError("Input must be 2‑D.")
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

    def run(self, data: np.ndarray | torch.Tensor) -> float:
        """Convenience wrapper returning a scalar float."""
        return float(self.forward(data).item())

# ----- FastBaseEstimator utilities -----
class FastBaseEstimator:
    """Evaluate a torch model for batches of inputs and observables."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        """Compute observables for each parameter set."""
        observables = list(observables) or [lambda x: x.mean()]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32)
                if inputs.ndim == 1:
                    inputs = inputs.unsqueeze(0)
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
        return results

class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""

    def evaluate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: Sequence[Sequence[float]],
                 *,
                 shots: int | None = None,
                 seed: int | None = None) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["Conv", "FastBaseEstimator", "FastEstimator"]
