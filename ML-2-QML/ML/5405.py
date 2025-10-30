"""Combined classical and hybrid estimator utilities.

This module extends the original FastBaseEstimator by adding:

* A **ClassicalSelfAttention** block that transforms input parameter sets
  before they are fed into a PyTorch model.
* A **HybridFunction**/**Hybrid** head that replaces a quantum expectation
  layer with a differentiable sigmoid.
* A **FastHybridEstimator** that supports batched evaluation, optional
  shot‑noise, and optional self‑attention.

The API is intentionally compatible with the original FastBaseEstimator
so that existing code can be upgraded with a single import change.
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


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
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
        return results


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


class ClassicalSelfAttention:
    """Simple multi‑head self‑attention implemented with PyTorch."""

    def __init__(self, embed_dim: int = 4, num_heads: int = 1):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = embed_dim ** -0.5

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        # Linear projections
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        # Scaled dot‑product attention
        scores = torch.softmax(query @ key.T * self.scale, dim=-1)
        return (scores @ value).numpy()


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation that mimics a quantum expectation head."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:  # type: ignore[override]
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None


class Hybrid(nn.Module):
    """Dense head that replaces a quantum circuit."""

    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)


class FastHybridEstimator(FastEstimator):
    """
    Combines a PyTorch model with an optional self‑attention transform
    and an optional hybrid head.

    Parameters
    ----------
    model : nn.Module
        Core neural network that produces feature vectors.
    attention : Optional[ClassicalSelfAttention]
        If given, the input parameters are first transformed by this
        block before being fed to *model*.
    hybrid_head : Optional[Hybrid]
        If given, the output of *model* is passed through this head to
        produce probabilities.
    """

    def __init__(
        self,
        model: nn.Module,
        attention: Optional[ClassicalSelfAttention] = None,
        hybrid_head: Optional[Hybrid] = None,
    ) -> None:
        super().__init__(model)
        self.attention = attention
        self.hybrid_head = hybrid_head

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        # Apply self‑attention if present
        if self.attention is not None:
            processed = [
                self.attention.run(np.array([0]), np.array([0]), np.array(p))
                for p in parameter_sets
            ]
            parameter_sets = [p.squeeze() for p in processed]
        raw = super().evaluate(observables, parameter_sets, shots=shots, seed=seed)
        # If a hybrid head is present, transform the last observable
        if self.hybrid_head is not None:
            # Assume last observable is the logits
            transformed = []
            for row in raw:
                logits = torch.tensor(row[-1], dtype=torch.float32).unsqueeze(0)
                probs = self.hybrid_head(logits).detach().cpu().numpy().tolist()
                transformed.append(probs)
            return transformed
        return raw


__all__ = [
    "FastBaseEstimator",
    "FastEstimator",
    "ClassicalSelfAttention",
    "HybridFunction",
    "Hybrid",
    "FastHybridEstimator",
]
