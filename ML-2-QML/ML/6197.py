"""Hybrid fast estimator for classical models with optional quantum hybrid layer.

This module implements a fast evaluation pipeline that can handle both pure classical
models and models that contain a quantum hybrid layer (e.g. a differentiable
quantum expectation layer).  The estimator supports deterministic evaluation,
shot‑noise injection and a convenient API that mirrors the original
FastBaseEstimator.

The implementation deliberately avoids any quantum dependencies so that it can
be used in environments where only PyTorch is available.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D tensor from a sequence of floats."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class HybridFastEstimator:
    """Fast evaluation of a classical or hybrid model.

    Parameters
    ----------
    model : torch.nn.Module
        The base model that produces raw logits or features.
    quantum_layer : torch.nn.Module | None
        Optional hybrid layer that wraps a quantum circuit.  The layer must
        accept a 1‑D tensor of shape (batch, features) and return a scalar
        per sample.  If ``None`` the model is evaluated directly.
    """

    def __init__(self, model: nn.Module, quantum_layer: nn.Module | None = None) -> None:
        self.model = model
        self.quantum_layer = quantum_layer

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate a list of observables for each parameter set.

        Parameters
        ----------
        observables
            Callables that map a tensor of model outputs to a scalar.
            If empty a default ``lambda out: out.mean()`` is used.
        parameter_sets
            Iterable of parameter vectors to feed to the model.
        shots
            If provided the result is perturbed with Gaussian noise
            whose variance is ``1/shots``.  This mimics shot‑noise.
        seed
            Random seed used when ``shots`` is not ``None``.

        Returns
        -------
        List[List[float]]
            Outer list over parameter sets, inner list over observables.
        """
        observables = list(observables) or [lambda out: out.mean()]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                raw = self.model(inputs)

                if self.quantum_layer is not None:
                    raw = self.quantum_layer(raw)

                row: List[float] = []
                for obs in observables:
                    val = obs(raw)
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

    def noise(self, results: List[List[float]], shots: int, seed: int | None = None) -> List[List[float]]:
        """Add Gaussian noise to deterministic results."""
        rng = np.random.default_rng(seed)
        return [[float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row] for row in results]

__all__ = ["HybridFastEstimator"]
