"""Classical quanvolution module and fast estimator.

The module mirrors the original `Quanvolution.py` but augments it with a
`FastBaseEstimator` that supports deterministic evaluation and optional
Gaussian shot noise.  The estimator is generic enough to be reused with
any PyTorch model that produces tensor outputs.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Coerce a 1‑D sequence into a batched 2‑D tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class QuanvolutionFilter(nn.Module):
    """Classical 2×2 filter implemented as a 2‑D convolution.

    The filter downsamples a single‑channel image by 2× and projects the
    resulting 4‑dimensional patch into a feature vector.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    """Classifier built atop the classical quanvolution filter."""

    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


class FastBaseEstimator:
    """Deterministic evaluator for PyTorch models.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.  It must accept a batched input tensor
        and return a tensor of arbitrary shape.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute scalar observables for each parameter set.

        Each observable is a callable that receives the model output
        and returns a scalar or a 1‑D tensor.  The results are aggregated
        into a list of lists.
        """
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
    """Adds Gaussian shot noise to deterministic outputs.

    The noise model is a normal distribution with variance 1/shots,
    which approximates the standard deviation of a binomial
    shot‑counting process.
    """

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = [
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "FastBaseEstimator",
    "FastEstimator",
]
