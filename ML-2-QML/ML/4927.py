"""Hybrid classical regression estimator with optional observables and Gaussian noise.

Provides a neural network that can be used to evaluate batches of inputs,
optionally applying user‑defined scalar observables.  The implementation
re‑uses ideas from the original EstimatorQNN, the regression model, and
the FastEstimator utilities.
"""

import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastEstimator:
    """Deterministic estimator with optional Gaussian shot noise."""
    def __init__(self, model: nn.Module):
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        import numpy as np
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


class EstimatorQNN(nn.Module):
    """Classical regression network with configurable depth and hidden size."""
    def __init__(self, input_dim: int = 2, hidden_size: int = 8, depth: int = 2, output_dim: int = 1):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(input_dim, hidden_size), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.Tanh()])
        layers.append(nn.Linear(hidden_size, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs).squeeze(-1)


__all__ = ["EstimatorQNN", "FastEstimator"]
