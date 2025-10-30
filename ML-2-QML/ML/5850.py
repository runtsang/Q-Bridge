"""Hybrid fast estimator for classical neural networks."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a list of parameter values to a 2‑D torch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """Evaluate a PyTorch model for many parameter sets and observables.

    The estimator is fully deterministic; optional Gaussian shot noise can
    be added to mimic quantum measurement statistics.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None) -> None:
        self.model = model.to(device or torch.device("cpu"))
        self.device = device or torch.device("cpu")

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Return a matrix of observable values.

        Parameters
        ----------
        observables: iterable of callables
            Each callable receives the model output tensor and returns a scalar
            or a tensor that will be reduced to a float.
        parameter_sets: sequence of sequences
            Each inner sequence contains the parameters for a forward pass.
        shots: optional int
            If provided, Gaussian noise with std = 1/√shots is added to each
            deterministic mean.
        seed: optional int
            Random seed for reproducibility of the noise.
        """
        if parameter_sets is None:
            parameter_sets = []
        if observables is None:
            observables = [lambda outputs: outputs.mean(dim=-1)]

        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
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

def EstimatorQNN() -> nn.Module:
    """Return a toy two‑layer regression network."""
    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(inputs)

    return EstimatorNN()

__all__ = ["FastBaseEstimator", "EstimatorQNN"]
