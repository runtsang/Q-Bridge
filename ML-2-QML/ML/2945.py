"""Hybrid self‑attention module for classical (PyTorch) execution.

The class mirrors the original SelfAttention logic but integrates the
FastBaseEstimator utilities for batch evaluation.  The quantum
counterpart is provided in the separate qml module.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Callable, Iterable, List, Sequence

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


class ClassicalSelfAttention(nn.Module):
    """Purely classical self‑attention block."""
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
    ) -> torch.Tensor:
        query = inputs @ rotation_params.reshape(self.embed_dim, -1)
        key = inputs @ entangle_params.reshape(self.embed_dim, -1)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs


class SelfAttentionHybrid:
    """Hybrid self‑attention that can run classically or delegate to a quantum circuit."""
    def __init__(self, embed_dim: int = 4, use_quantum: bool = False) -> None:
        self.embed_dim = embed_dim
        self.use_quantum = use_quantum
        self.classical = ClassicalSelfAttention(embed_dim)
        self.estimator = FastBaseEstimator(self.classical)

    def run(
        self,
        inputs: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        if self.use_quantum:
            raise RuntimeError("Quantum execution requires the qml module.")
        inputs_t = torch.as_tensor(inputs, dtype=torch.float32)
        rotation_t = torch.as_tensor(rotation_params, dtype=torch.float32)
        entangle_t = torch.as_tensor(entangle_params, dtype=torch.float32)
        out = self.classical(inputs_t, rotation_t, entangle_t)
        return out.detach().cpu().numpy()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Delegate to FastBaseEstimator for batch evaluation."""
        return self.estimator.evaluate(observables, parameter_sets)


__all__ = ["SelfAttentionHybrid"]
