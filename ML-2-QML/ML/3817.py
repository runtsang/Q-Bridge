"""Hybrid estimator for classical neural networks with optional self‑attention and shot noise."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Helper: self‑attention preprocessing
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    """Simple dot‑product self‑attention implemented with PyTorch."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(x)
        key   = self.key_proj(x)
        value = self.value_proj(x)
        scores = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, value)

# --------------------------------------------------------------------------- #
# Utility to ensure batched input
# --------------------------------------------------------------------------- #
def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

# --------------------------------------------------------------------------- #
# Main estimator
# --------------------------------------------------------------------------- #
class HybridBaseEstimator:
    """
    Evaluate a PyTorch model (optionally wrapped with self‑attention) for many
    parameter sets and observables.  Observables are callables that map a
    model output to a scalar (or tensor).
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        attention: bool = False,
        attention_dim: Optional[int] = None,
    ) -> None:
        self.model = model
        if attention:
            if attention_dim is None:
                raise ValueError("attention_dim must be specified when attention=True")
            self.attention = ClassicalSelfAttention(attention_dim)
        else:
            self.attention = None

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Parameters
        ----------
        observables: iterable of callables
            Each callable receives the model output and returns a scalar
            (or a tensor convertible to float).
        parameter_sets: sequence of parameter value sequences
            Each sequence is fed to the model as a batch of inputs.
        shots, seed: optional
            When provided, Gaussian noise with variance ~1/shots is added to
            each observable to emulate measurement noise.
        """
        observables = list(observables) or [lambda x: x.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        if self.attention:
            self.attention.eval()

        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                # Optional preprocessing
                if self.attention:
                    inputs = self.attention(inputs)
                outputs = self.model(inputs)

                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)

        # Add shot noise if requested
        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [
                    float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
                ]
                noisy.append(noisy_row)
            return noisy

        return results


__all__ = ["HybridBaseEstimator"]
