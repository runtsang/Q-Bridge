"""Hybrid classical estimator combining convolutional feature extraction, fully‑connected regression, and optional Gaussian shot noise.

The implementation follows the EstimatorQNN anchor but expands on the Conv, QuantumNAT, and FastEstimator examples:
- A 2‑D convolution with a user‑configurable threshold (inspired by Conv.py).
- A small fully‑connected backbone (inspired by EstimatorQNN.py).
- Optional shot‑noise emulation via a Gaussian perturbation (inspired by FastEstimator.py).
- A convenient ``evaluate`` method that mirrors the quantum estimator interface, returning expectation‑value‑like scalars for arbitrary observables.

This module is fully classical (NumPy, PyTorch) and can be dropped into any PyTorch training pipeline.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence into a 2‑D torch tensor (batch, features)."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class EstimatorQNN__gen223(nn.Module):
    """
    Classical estimator that mimics a quantum‑inspired regression model.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square convolution kernel, also determines the input
        dimensionality (kernel_size²).
    hidden_dim : int, default 8
        Size of the hidden layer in the fully‑connected head.
    threshold : float, default 0.0
        Threshold applied after the convolution before the sigmoid
        activation (see Conv.py).
    shots : int | None, default None
        If given, the model will return noisy estimates by adding
        Gaussian noise with variance 1/shots.
    seed : int | None, default None
        Random seed for reproducibility of the noise.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        hidden_dim: int = 8,
        threshold: float = 0.0,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size, bias=True)
        self.fc = nn.Sequential(
            nn.Linear(kernel_size * kernel_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.shots = shots
        self.seed = seed
        if shots is not None:
            self.rng = np.random.default_rng(seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass.

        Accepts either a 2‑D tensor ``(batch, 2)`` or a full 4‑D tensor.
        """
        if x.dim() == 2:
            x = x.view(-1, 1, self.kernel_size, self.kernel_size)
        conv_out = self.conv(x)
        activations = torch.sigmoid(conv_out - self.threshold)
        flat = activations.view(x.size(0), -1)
        return self.fc(flat)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Evaluate the model for a collection of input vectors and observables.

        Parameters
        ----------
        observables : Iterable[Callable[[torch.Tensor], torch.Tensor | float]]
            Callables that map the model output to a scalar.  If empty a
            default that returns the mean of the last dimension is used.
        parameter_sets : Sequence[Sequence[float]]
            List of input vectors (each of length ``kernel_size**2``).

        Returns
        -------
        List[List[float]]
            A matrix of shape ``(len(parameter_sets), len(observables))``.
        """
        observables = list(observables) or [lambda o: o.mean(dim=-1)]
        results: List[List[float]] = []

        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self(inputs)
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
                    scalar = float(value.mean().cpu()) if isinstance(value, torch.Tensor) else float(value)
                    row.append(scalar)
                results.append(row)

        # Add Gaussian shot noise if requested
        if self.shots is not None:
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [float(self.rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy

        return results


__all__ = ["EstimatorQNN__gen223"]
