from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Callable, Iterable, List, Sequence

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridFastEstimator:
    """Hybrid estimator that wraps a PyTorch model and evaluates observables.

    Parameters
    ----------
    model : nn.Module
        A PyTorch module mapping input parameters to an output tensor.
    shots : int | None, optional
        If supplied, Gaussian shot noise is added to deterministic outputs.
    seed : int | None, optional
        Random seed for reproducibility of shot noise.
    """

    def __init__(self, model: nn.Module, *, shots: int | None = None, seed: int | None = None) -> None:
        self.model = model
        self.shots = shots
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate the model for each parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of callable functions that map the model output to a scalar.
        parameter_sets
            Sequence of parameter vectors to evaluate.

        Returns
        -------
        List[List[float]]
            Nested list of observable values for each parameter set.
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

        if self.shots is None:
            return results

        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [
                float(self.rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


def EstimatorQNN() -> nn.Module:
    """Return a simple regression neural network."""
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


def SamplerQNN() -> nn.Module:
    """Return a simple classification neural network producing probabilities."""
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return torch.softmax(self.net(inputs), dim=-1)

    return SamplerModule()


__all__ = ["HybridFastEstimator", "EstimatorQNN", "SamplerQNN"]
