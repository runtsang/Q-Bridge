"""Hybrid fully‑connected layer implemented purely with PyTorch.

The class mirrors the structure of EstimatorQNN but augments it with
batch‑wise evaluation, observable handling and optional Gaussian shot
noise, following the spirit of FastBaseEstimator.  The API is
compatible with the original FCL example: ``HybridFCL().evaluate(...)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

# Type alias for scalar observables that can be applied to the network output.
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridFCL(nn.Module):
    """
    Classical feed‑forward network that supports batched evaluation of
    arbitrary scalar observables and optional Gaussian shot noise.

    Parameters
    ----------
    n_features : int, default 1
        Size of the input feature vector.
    hidden_sizes : Sequence[int], default (8, 4)
        Sizes of the hidden layers, mirroring EstimatorQNN.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_sizes: Sequence[int] = (8, 4),
    ) -> None:
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_dim, h), nn.Tanh()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Evaluate the network for a collection of parameter sets and observables.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Callable objects that map the network output to a scalar.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter sequences (each matching the network input size).
        shots : Optional[int], default None
            If provided, Gaussian noise with variance 1/shots is added
            to each observable value.
        seed : Optional[int], default None
            Random seed for reproducible noise.

        Returns
        -------
        List[List[float]]
            Outer list indexed by parameter set, inner list by observable.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self(inputs)
                row: List[float] = []
                for observable in observables:
                    val = observable(outputs)
                    scalar = float(val.mean().cpu()) if isinstance(val, torch.Tensor) else float(val)
                    row.append(scalar)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["HybridFCL"]
