"""Hybrid fully‑connected layer with classical evaluation and optional shot‑noise emulation.

The class is compatible with the original FCL anchor: a function ``FCL()`` returns an instance
that exposes a ``run`` method for single‑parameter inference and an ``evaluate`` method for
batch evaluation of arbitrary scalar observables.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Callable, Iterable, List, Sequence

# Alias for observables that map a tensor to a scalar
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a 2‑D torch tensor for batch processing."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FCLHybrid(nn.Module):
    """
    Classical fully‑connected layer that supports:
    * deterministic forward pass
    * optional Gaussian shot‑noise (mimicking quantum measurements)
    * batch evaluation of arbitrary scalar observables (FastBaseEstimator style)
    """

    def __init__(self, n_features: int = 1, noise_shots: int | None = None, seed: int | None = None) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.noise_shots = noise_shots
        self.seed = seed

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        """Deterministic forward pass returning a 1‑D tensor of the expectation."""
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return torch.tanh(self.linear(values)).mean(dim=0)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Return a NumPy array of the expectation value, optionally adding shot noise."""
        out = self.forward(thetas)
        if self.noise_shots is None:
            return out.detach().numpy()
        rng = np.random.default_rng(self.seed)
        noisy = rng.normal(out.item(), max(1e-6, 1 / self.noise_shots))
        return np.array([noisy])

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Batch evaluation of multiple parameter sets and observables.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Functions mapping the model output tensor to a scalar.
        parameter_sets : Sequence[Sequence[float]]
            Each inner sequence contains the parameters for one forward pass.

        Returns
        -------
        List[List[float]]
            A matrix where each row corresponds to a parameter set and each column
            to an observable.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.linear(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results


def FCL() -> FCLHybrid:
    """Convenience factory matching the original FCL anchor."""
    return FCLHybrid()
