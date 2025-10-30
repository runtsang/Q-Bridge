"""Hybrid quanvolution model – classical implementation.

This module implements a deterministic classifier that mirrors the
original `QuanvolutionFilter` and `QuanvolutionClassifier` but adds a
shot‑noise wrapper inspired by the FastEstimator pattern.  The class is
fully compatible with PyTorch and can be used as a drop‑in replacement
in existing pipelines.

Key features
------------
* 2×2 convolutional filter → 4‑channel feature map.
* Linear head for 10‑class classification.
* `evaluate` method that accepts a list of scalar observables and
  optional shot‑noise simulation.
"""

from __future__ import annotations

import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class QuanvolutionHybridModel(nn.Module):
    """Deterministic quanvolution classifier with optional shot‑noise.

    The forward pass reproduces the behaviour of the original
    ``QuanvolutionFilter`` and ``QuanvolutionClassifier``.  The
    ``evaluate`` method implements a lightweight FastEstimator
    interface: it evaluates a batch of parameter sets and can add
    Gaussian shot‑noise to the outputs, mimicking quantum sampling.
    """

    def __init__(self, n_features: int = 4 * 14 * 14) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.linear = nn.Linear(n_features, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        flat = features.view(x.size(0), -1)
        logits = self.linear(flat)
        return F.log_softmax(logits, dim=-1)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate the model on a list of parameter sets.

        Parameters
        ----------
        observables:
            Iterable of callables that accept the model output and return
            a scalar tensor or float.
        parameter_sets:
            Sequence of sequences of float values.  Each inner sequence
            is treated as a single input sample.
        shots:
            If provided, Gaussian noise with variance 1/shots is added to
            each scalar output to emulate quantum shot noise.
        seed:
            Optional RNG seed for reproducibility.
        """
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.forward(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    row.append(float(val))
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy_results: List[List[float]] = []
        for row in results:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy_results.append(noisy_row)
        return noisy_results


__all__ = ["QuanvolutionHybridModel"]
