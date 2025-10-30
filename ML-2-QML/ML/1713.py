"""
Hybrid estimator that evaluates a PyTorch neural network with optional shot noise.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Sequence, Callable

# A callable that maps model outputs to a scalar (or a tensor that can be reduced).
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of floats to a 2‑D torch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastHybridEstimator:
    """
    Evaluate a PyTorch model for batched inputs with optional Gaussian shot noise.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network to be evaluated.

    Notes
    -----
    The API mirrors the original FastEstimator, but the implementation
    combines the base estimator with shot‑noise simulation in a single class.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Compute predictions for each parameter set and observable.

        Parameters
        ----------
        observables
            Callables that map the model output to a scalar.
        parameter_sets
            Iterable of input parameter sequences.
        shots
            If provided, adds Gaussian noise with variance 1/shots.
        seed
            RNG seed for the noise generator.

        Returns
        -------
        List[List[float]]
            Rows of observable values for each parameter set.
        """
        obs = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in obs:
                    value = observable(outputs)
                    scalar = float(value.mean().cpu()) if isinstance(value, torch.Tensor) else float(value)
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


__all__ = ["FastHybridEstimator"]
