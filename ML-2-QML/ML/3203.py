"""Fast hybrid estimator for classical neural networks with optional shot noise.

This module extends the lightweight FastBaseEstimator by adding a
Gaussian noise layer to mimic measurement uncertainty.  The public API
matches the quantum version so that a user can swap the backend
without changing the surrounding code.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastHybridEstimator:
    """Evaluate a PyTorch model over many parameter sets.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.  It is assumed to accept a
        batch of inputs and return a batch of outputs.
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
        Compute scalar observables for each parameter set.

        The method first runs the model in evaluation mode, then
        applies each observable to the outputs.  If ``shots`` is
        provided, Gaussian noise with variance 1/shots is added to
        each mean value to emulate measurement uncertainty.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output tensor and
            returns a scalar tensor or float.
        parameter_sets : sequence of sequences
            Batches of input values to feed to the model.
        shots : int, optional
            Number of measurement shots to simulate.  When ``None``
            (default) the estimator is deterministic.
        seed : int, optional
            Random seed for the Gaussian noise.

        Returns
        -------
        List[List[float]]
            A 2‑D list where the outer dimension corresponds to
            parameter sets and the inner dimension to observables.
        """
        if not isinstance(observables, Iterable):
            raise TypeError("observables must be an iterable of callables")

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
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)

        return noisy


__all__ = ["FastHybridEstimator"]
