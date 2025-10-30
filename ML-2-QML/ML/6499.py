"""Hybrid sampler network with classical and quantum evaluation capabilities.

This module defines a `SamplerQNN` class that can be instantiated in a purely
classical context.  It uses PyTorch to construct a small feed‑forward network
with an optional dropout layer to encourage robustness.  The class also
provides an `evaluate` method that accepts arbitrary scalar observables
(callables) and a list of parameter sets, returning the computed values in a
batch‑friendly manner.  Optional Gaussian shot noise can be added to mimic
quantum sampling statistics.

The design mirrors the structure of the original `SamplerQNN` and `FastBaseEstimator`
seeds while extending them with batch processing, dropout, and noise support.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class SamplerQNN(nn.Module):
    """
    Classical sampler network.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input parameter vector.
    hidden_dim : int, default 4
        Size of the hidden layer.
    dropout : float, default 0.0
        Dropout probability applied after the hidden layer.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.extend(
            [
                nn.Linear(hidden_dim, input_dim),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return a probability distribution over the output dimension."""
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate the network on a batch of parameter sets.

        Parameters
        ----------
        observables : iterable of callables
            Each callable accepts the network output tensor and returns a
            scalar or a tensor that can be reduced to a scalar.
        parameter_sets : sequence of sequences
            Each inner sequence contains the parameters for a single forward pass.
        shots : int, optional
            If provided, Gaussian noise with variance 1/shots is added to each
            observable value to emulate quantum shot noise.
        seed : int, optional
            Random seed for reproducible noise.

        Returns
        -------
        List[List[float]]
            Nested list of observable values for each parameter set.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy

        return results


__all__ = ["SamplerQNN"]
