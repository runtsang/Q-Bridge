"""Hybrid estimator that unifies classical neural‑net and quantum circuit evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D tensor with a batch dimension."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridBaseEstimator:
    """Classical estimator based on a PyTorch `nn.Module`.

    Parameters
    ----------
    model : nn.Module
        Neural network to evaluate.
    dropout_prob : float, optional
        Dropout probability applied to the output during
        evaluation.  Useful when simulating measurement noise.
    """

    def __init__(self, model: nn.Module, dropout_prob: float = 0.0) -> None:
        self.model = model
        self.dropout_prob = dropout_prob

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """Evaluate observables for many parameter sets.

        Parameters
        ----------
        observables : iterable
            Functions that map the model output to a scalar.
        parameter_sets : sequence
            Iterable of parameter vectors.
        shots : int, optional
            If provided, Gaussian noise with variance 1/shots is added
            to the mean value of each observable.
        seed : int, optional
            Random seed for reproducible noise.

        Returns
        -------
        List[List[float]]
            Nested list with shape
            ``(len(parameter_sets), len(observables))``.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)

                if self.dropout_prob > 0.0:
                    outputs = torch.nn.functional.dropout(
                        outputs, p=self.dropout_prob, training=False
                    )

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
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


# Backwards‑compatibility aliases
FastBaseEstimator = HybridBaseEstimator
FastEstimator = HybridBaseEstimator
__all__ = ["HybridBaseEstimator", "FastBaseEstimator", "FastEstimator"]
