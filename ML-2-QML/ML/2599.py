"""Hybrid estimator combining PyTorch evaluation and optional classical self‑attention.

This module extends the lightweight ``FastBaseEstimator`` from the original
anchor file, adding a modular attention layer that can be swapped in or out
without changing the public API.  The class retains the ``evaluate`` method
signature so existing pipelines continue to work, while exposing an
``apply_attention`` hook that runs a classical self‑attention block on the
input parameters before they are fed into the model.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

# Base estimator copied from the anchor with slight refactor
class _BaseEstimator:
    """Base class for deterministic estimators."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate the model for each parameter set and observable."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = self._to_tensor_batch(params)
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
        return results

    @staticmethod
    def _to_tensor_batch(values: Sequence[float]) -> torch.Tensor:
        t = torch.as_tensor(values, dtype=torch.float32)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        return t

# Classical self‑attention helper
class _ClassicalSelfAttention:
    """Simple multi‑head self‑attention implemented in PyTorch."""

    def __init__(self, embed_dim: int = 4, heads: int = 1) -> None:
        self.embed_dim = embed_dim
        self.heads = heads
        self.scale = np.sqrt(embed_dim)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Compute attention output."""
        q = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        k = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        v = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(q @ k.T / self.scale, dim=-1)
        return (scores @ v).numpy()

def _build_classical_attention(embed_dim: int = 4) -> _ClassicalSelfAttention:
    """Factory returning a classical attention instance."""
    return _ClassicalSelfAttention(embed_dim=embed_dim)

class HybridEstimator(_BaseEstimator):
    """Hybrid estimator that optionally applies a classical self‑attention
    transform before evaluating a PyTorch model.
    """

    def __init__(
        self,
        model: nn.Module,
        attention: Optional[_ClassicalSelfAttention] = None,
    ) -> None:
        super().__init__(model)
        self.attention = attention or _build_classical_attention()

    def apply_attention(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Apply the internal attention mechanism to the inputs."""
        return self.attention.run(rotation_params, entangle_params, inputs)

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        use_attention: bool = False,
    ) -> List[List[float]]:
        """Evaluate the model, optionally applying attention to the inputs."""
        if use_attention:
            rotated = []
            for params in parameter_sets:
                # Assume first 8 params for rotation/entangle (2 * embed_dim * 4)
                rot = params[:8]
                ent = params[8:16]
                inp = params[16:]
                attn_out = self.apply_attention(rot, ent, inp)
                rotated.append(attn_out)
            # Flatten the attention output to match model input shape
            parameter_sets = [list(p) for p in rotated]
        return super().evaluate(observables, parameter_sets)

__all__ = ["HybridEstimator"]
