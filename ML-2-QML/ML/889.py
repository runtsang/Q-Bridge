"""Hybrid estimator that couples a PyTorch surrogate model with a quantum circuit backend."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Protocol

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Helper types
# --------------------------------------------------------------------------- #
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D tensor of shape (batch, 1) for any 1‑D input sequence."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor.unsqueeze_(0)
    return tensor


# --------------------------------------------------------------------------- #
# Hybrid estimator
# --------------------------------------------------------------------------- #
class FastHybridEstimator:
    """
    A lightweight estimator that evaluates either a quantum backend or a
    classical surrogate model.

    Parameters
    ----------
    model : nn.Module
        PyTorch model that acts as a surrogate for the quantum circuit.
    backend : Optional[Callable[[Sequence[float]], torch.Tensor]]
        Optional callable that maps a parameter vector to a model output.  If
        ``None`` the surrogate model is used directly.  The backend must return
        a torch tensor of shape ``(output_dim,)``.
    """

    def __init__(
        self,
        model: nn.Module,
        backend: Optional[Callable[[Sequence[float]], torch.Tensor]] = None,
    ) -> None:
        self.model = model
        self.backend = backend

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Compute expectation values for each observable and parameter set.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Functions that map a model output tensor to a scalar.
        parameter_sets : Sequence[Sequence[float]]
            List of parameter vectors.
        shots : Optional[int]
            If provided, add Gaussian shot noise with variance ``1/shots``.
        seed : Optional[int]
            Random seed for noise generation.

        Returns
        -------
        List[List[float]]
            Nested list of scalar results.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                batch = _ensure_batch(params)
                if self.backend is None:
                    outputs = self.model(batch).squeeze(0)
                else:
                    outputs = self.backend(params)

                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
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


__all__ = ["FastHybridEstimator"]
