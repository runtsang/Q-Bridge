"""
ML implementation of a hybrid fully‑connected quantum‑layer estimator.
Combines the lightweight FastBaseEstimator pattern with a classical
neural‑network surrogate for the quantum layer.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FCLHybridEstimator(nn.Module):
    """
    Classical surrogate for a fully‑connected quantum layer.

    Parameters
    ----------
    n_features : int
        Number of input features (equivalent to number of qubits).
    bias : bool
        Whether to include a bias term.
    """

    def __init__(self, n_features: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=bias)

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        """Return the network output for a single set of parameters."""
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return torch.tanh(self.linear(values))

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Mimic the quantum ``run`` interface."""
        return self.forward(thetas).mean(dim=0).detach().numpy()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Evaluate a list of observables for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives a batch of outputs and returns a
            scalar or tensor that can be reduced to a float.
        parameter_sets : sequence of sequences
            Batches of parameters to evaluate.
        shots : int, optional
            Inject Gaussian shot noise with variance 1/shots.
        seed : int, optional
            Random seed for reproducible noise.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                batch = _ensure_batch(params)
                outputs = self(batch)  # shape [batch, 1]
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
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


def FCL(n_features: int = 1, bias: bool = True) -> FCLHybridEstimator:
    return FCLHybridEstimator(n_features, bias)


__all__ = ["FCLHybridEstimator", "FCL"]
