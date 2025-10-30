"""Hybrid classifier with classical convolution and a quantum-inspired feature extractor.

This module implements the same public API as the original `QuanvolutionClassifier`
but replaces the quantum filter with a lightweight classical surrogate.  The
surrogate mimics the dimensionality of the quantum kernel and can be swapped
with a true quantum module without changing the surrounding training code.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
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
    """Adds optional Gaussian shot noise to the deterministic estimator."""
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
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


class ClassicalQuantumFeatureExtractor(nn.Module):
    """Classical surrogate for the quantum kernel."""
    def __init__(self, input_dim: int, feature_dim: int = 4):
        super().__init__()
        self.linear = nn.Linear(input_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class QuanvolutionHybridClassifier(nn.Module):
    """
    Classical hybrid classifier that combines a 2‑D convolution with a
    quantum‑inspired feature extractor.

    The design mirrors the original `QuanvolutionClassifier` but replaces the
    quantum filter with `ClassicalQuantumFeatureExtractor`.  This allows
    training and evaluation on CPU while still offering a plug‑in point
    for a true quantum implementation.
    """
    def __init__(self, n_channels: int = 1, n_classes: int = 10) -> None:
        super().__init__()
        self.conv = nn.Conv2d(n_channels, 4, kernel_size=2, stride=2)
        self.qf_extractor = ClassicalQuantumFeatureExtractor(
            input_dim=4 * 14 * 14,
            feature_dim=4 * 14 * 14,
        )
        self.linear = nn.Linear(4 * 14 * 14 * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)            # shape: [B,4,14,14]
        flat = features.view(features.size(0), -1)
        q_features = self.qf_extractor(flat)  # quantum‑surrogate features
        combined = torch.cat([flat, q_features], dim=1)
        logits = self.linear(combined)
        return F.log_softmax(logits, dim=-1)


__all__ = [
    "QuanvolutionHybridClassifier",
    "FastEstimator",
    "FastBaseEstimator",
    "ClassicalQuantumFeatureExtractor",
]
