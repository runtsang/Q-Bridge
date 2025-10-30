"""Hybrid estimator that extends FastBaseEstimator with feature extraction and normalization."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D float tensor from a sequence.  The function accepts both 1‑D and 2‑D
    input and guarantees a batch dimension for the output."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FeatureExtractor(nn.Module):
    """Simple feature‑extraction network that can be used with the base estimator."""
    def __init__(self, in_dim: int, hidden_dims: Sequence[int] = (64, 32), out_dim: int | None = None):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            prev_dim = hdim
        if out_dim is not None:
            layers.append(nn.Linear(prev_dim, out_dim))
            prev_dim = out_dim
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class QuantumHybridEstimator:
    """
    Lightweight estimator that extends the original FastBaseEstimator with
    configurable feature extraction and batch‑level normalisation.  Shots‑noise
    can be added in a post‑processing step.
    """
    def __init__(
        self,
        model: nn.Module,
        *,
        feature_extractor: nn.Module | None = None,
        normalizer: nn.Module | None = None,
    ) -> None:
        self.model = model
        self.feature_extractor = feature_extractor or nn.Identity()
        self.normalizer = normalizer or nn.Identity()

    def _preprocess(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply feature extraction and normalisation before feeding to the model."""
        x = self.feature_extractor(inputs)
        x = self.normalizer(x)
        return x

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate scalar observables over a batch of parameter sets.  If *shots* is
        provided, Gaussian shot‑noise is added to each mean value.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                prepped = self._preprocess(inputs)
                outputs = self.model(prepped)
                row: List[float] = []
                for observable in observables:
                    val = observable(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
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

__all__ = ["QuantumHybridEstimator"]
