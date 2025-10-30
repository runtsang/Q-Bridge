"""Hybrid estimator that evaluates a PyTorch model (classical or hybrid) with optional shot noise.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Classical quanvolution filter
# --------------------------------------------------------------------------- #
class ClassicalQuanvolutionFilter(nn.Module):
    """
    A lightweight 2×2 patch filter implemented as a single Conv2d layer.
    Mirrors the behaviour of the original ``QuanvolutionFilter`` but with
    explicit parameters that can be inspected or replaced.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Flatten the convolution output into a 2‑D feature vector."""
        features = self.conv(x)
        return features.view(x.size(0), -1)

# --------------------------------------------------------------------------- #
# Hybrid classifier that uses the quanvolution filter
# --------------------------------------------------------------------------- #
class HybridClassifier(nn.Module):
    """
    Combines the classical quanvolution filter with a linear head.
    The architecture is identical to the original example but exposed as a
    reusable module for the estimator.
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = ClassicalQuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

# --------------------------------------------------------------------------- #
# Hybrid estimator
# --------------------------------------------------------------------------- #
class HybridEstimator:
    """
    Evaluate a PyTorch model for a collection of parameter sets and
    observables.  Observables are callables that map the model output to a
    scalar or a tensor that can be reduced to a scalar.  The estimator
    optionally injects Gaussian shot noise to emulate quantum measurement
    statistics.
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @staticmethod
    def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output and returns a scalar or
            a tensor that can be reduced to a scalar.
        parameter_sets : sequence of parameter lists
            Every list is fed to the model as a single batch element.
        shots : int | None
            If provided, Gaussian noise with variance 1/shots is added to each
            observable value to emulate finite‑shot statistics.
        seed : int | None
            RNG seed for reproducible noise.

        Returns
        -------
        List[List[float]]
            A 2‑D list where each row corresponds to a parameter set and each
            column to an observable.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = self._ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
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

__all__ = ["HybridEstimator", "HybridClassifier", "ClassicalQuanvolutionFilter"]
