"""Hybrid estimator that extends FastBaseEstimator with a feature extractor and optional shot noise.

The estimator accepts a PyTorch neural network that maps input parameter vectors to a feature vector,
which is then used directly as the model output. It supports batched inference and optional Gaussian
shot noise to mimic stochastic quantum measurements. The public API mirrors the original FastBaseEstimator
class but adds a ``feature_extractor`` argument and a ``shots`` parameter.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastHybridEstimator:
    """Evaluate a neural‑network feature extractor with optional shot noise.

    Parameters
    ----------
    model : nn.Module
        A pure‑trainable PyTorch model that maps input parameters to a feature vector.
    shots : int | None, optional
        Number of simulated shots. If ``None`` the estimator is deterministic.
    seed : int | None, optional
        Random seed for the noise generator when shots are used.
    """

    __slots__ = ("model", "shots", "seed", "rng")

    def __init__(self, model: nn.Module, *, shots: int | None = None, seed: int | None = None) -> None:
        self.model = model
        self.shots = shots
        self.seed = seed
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a 2‑D list of observable values for each parameter set.

        The observable functions receive the model output and may return a scalar tensor
        or a Python scalar.  When ``shots`` is set, Gaussian noise with standard
        deviation ``1 / sqrt(shots)`` is added independently to each observable.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)

                if self.shots is not None:
                    row = [float(v + self.rng.normal(0, 1 / np.sqrt(self.shots))) for v in row]

                results.append(row)

        return results


__all__ = ["FastHybridEstimator"]
