from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Callable, Sequence

class HybridFullyConnectedLayer(nn.Module):
    """
    Classical surrogate of a fully‑connected quantum layer.
    Supports batched evaluation and optional observable mapping.
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compute the expectation (mean tanh output) for a single parameter set.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        out = torch.tanh(self.linear(values))
        expectation = out.mean(dim=0).item()
        return np.array([expectation])

class HybridEstimator:
    """
    Evaluates a model on a list of parameter sets and observables.
    Adds optional Gaussian shot noise to mimic quantum measurement statistics.
    """
    def __init__(self, model: nn.Module):
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    row.append(val)
                results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy = [[float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                 for row in results]
        return noisy

def FCL(n_features: int = 1) -> HybridFullyConnectedLayer:
    """Return a hybrid fully connected layer instance."""
    return HybridFullyConnectedLayer(n_features)

__all__ = ["HybridFullyConnectedLayer", "HybridEstimator", "FCL"]
