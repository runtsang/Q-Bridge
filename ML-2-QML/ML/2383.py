"""Hybrid estimator that can evaluate classical neural networks and quantum circuits.

This module defines FastHybridEstimator that extends FastEstimator with
additional support for probabilistic sampling and a small sampler network.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Callable, Iterable, List, Sequence, Union

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class SamplerQNN(nn.Module):
    """Simple neural network that outputs a probability distribution over two classes."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.softmax(self.net(inputs), dim=-1)

class FastHybridEstimator:
    """Evaluate a PyTorch model for batches of inputs and observables.

    Supports optional Gaussian shot noise to emulate finite‑shot sampling.
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] = (),
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Parameters
        ----------
        observables
            Callables that map a model output tensor to a scalar. If None,
            the mean of the output tensor is used.
        parameter_sets
            Iterable of parameter vectors that are fed to the model.
        shots
            If provided, Gaussian noise with variance 1/shots is added to each
            scalar result to mimic finite‑shot estimation.
        seed
            Random seed for reproducible noise generation.
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

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["FastHybridEstimator", "SamplerQNN"]
