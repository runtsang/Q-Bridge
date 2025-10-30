"""Hybrid estimator for pure PyTorch models with optional Gaussian shot noise.

Provides a unified API for evaluating both classical neural networks and
hybrid CNN‑linear architectures (e.g. QFCModelClassic) on batches of
parameter sets.  Noise is added only on request, mirroring the
FastEstimator behaviour from the original scaffold.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
Observable = Union[ScalarObservable, Callable[[torch.Tensor], torch.Tensor]]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a list of floats into a 2‑D float32 tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridBaseEstimator:
    """Evaluate a PyTorch model for a collection of input parameter sets.

    Parameters
    ----------
    model : nn.Module
        A pure PyTorch model (e.g. ``QFCModelClassic`` or any custom Net).
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Observable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a matrix of observable values.

        Each row corresponds to a parameter set; each column to an observable.
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
                    val = observable(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results


class HybridEstimator(HybridBaseEstimator):
    """Same as :class:`HybridBaseEstimator` but injects Gaussian shot noise."""

    def evaluate(
        self,
        observables: Iterable[Observable],
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
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


# ----------------------------------------------------------------------
# Classical hybrid model inspired by Quantum‑NAT
# ----------------------------------------------------------------------
class QFCModelClassic(nn.Module):
    """CNN → FC → BatchNorm architecture (four‑dimensional output)."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)


__all__ = ["HybridBaseEstimator", "HybridEstimator", "QFCModelClassic"]
