"""Unified estimator for PyTorch models.

This module extends the original FastBaseEstimator logic to support:
- batched evaluation of arbitrary observable callables.
- optional Gaussian shot noise with automatic scaling.
- a flexible interface that can be used as a drop‑in for any nn.Module
  including hybrid quantum‑classical wrappers.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

class UnifiedBaseEstimator:
    """Estimator that evaluates a PyTorch model on a batch of parameters.

    Parameters
    ----------
    model:
        A PyTorch ``nn.Module`` that accepts a 2‑D tensor of shape
        ``(batch_size, input_dim)`` and returns a 2‑D tensor of shape
        ``(batch_size, output_dim)``.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @staticmethod
    def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
        """Return a tensor with shape ``(1, n)`` for a single‑sample batch."""
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Run the model on each parameter set and compute scalar observables.

        Parameters
        ----------
        observables:
            Callables that map the model output to a scalar (or a tensor that
            will be reduced to a scalar).  If omitted, the mean over the last
            dimension of the output is used.
        parameter_sets:
            Iterable of parameter sequences; each sequence is fed as a
            single input to the model.
        shots:
            If provided, Gaussian noise with ``var=1/shots`` is added to each
            observable value to emulate measurement statistics.
        seed:
            Random seed for reproducibility of the synthetic shot noise.
        Returns
        -------
        List[List[float]]:
            A list of rows, one per parameter set.  Each row contains the
            evaluated observable values in the order supplied.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = self._ensure_batch(params)
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
            noisy_row = [rng.normal(mean, max(1e-6, 1 / shots)) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["UnifiedBaseEstimator"]
