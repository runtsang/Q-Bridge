"""Hybrid classical estimator built on FastBaseEstimator and a fully connected PyTorch layer."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Sequence, Callable

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of floats to a 2‑D torch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FullyConnectedLayer(nn.Module):
    """Simple 1‑hidden‑unit fully‑connected layer used as a stand‑in for a quantum layer."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x))

class FastBaseEstimatorGen326:
    """Hybrid estimator that evaluates a PyTorch model over a batch of parameters
    and optionally adds Gaussian shot noise to replicate quantum‑shot effects."""
    def __init__(self, model: nn.Module | None = None) -> None:
        # Default to a 1‑feature fully connected layer if no model is supplied.
        self.model = model if model is not None else FullyConnectedLayer()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] = (),
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate each observable for every parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output tensor and returns a scalar.
        parameter_sets : sequence of parameter sequences
            Each inner sequence corresponds to one evaluation.
        shots : int, optional
            If supplied, Gaussian noise with variance 1/shots is added to each result.
        seed : int, optional
            Random seed for reproducibility of shot noise.
        """
        if observables is None:
            observables = [lambda out: out.mean(dim=-1)]
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
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["FastBaseEstimatorGen326"]
