"""Hybrid estimator combining classical PyTorch neural nets and optional shot noise.

The class FastBaseEstimator evaluates a PyTorch model on batches of parameters
and aggregates the outputs using user supplied scalar observables.  It further
provides a lightweight classical self‑attention block that can be plugged into
the model, mirroring the interface of the quantum self‑attention implementation
present in the repository.  The estimator can optionally add Gaussian shot
noise to mimic measurement uncertainty, enabling fair comparison with the
quantum side.

The design follows a combination scaling paradigm: the classical part scales
with the number of model parameters and batch size, while the quantum part
scales with the number of shots and circuit depth.  Both halves expose a
compatible API (``evaluate``) so that a single script can run either
classical or quantum experiments interchangeably.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a list of scalars into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class ClassicalSelfAttention(nn.Module):
    """
    Minimal self‑attention module that mimics the interface of the quantum
    self‑attention circuit.  It accepts rotation and entangle parameters
    (flattened) and an input batch and returns the attended outputs.
    """

    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        rot_mat = rotation_params.reshape(self.embed_dim, -1)
        ent_mat = entangle_params.reshape(self.embed_dim, -1)

        query = torch.matmul(inputs, rot_mat)
        key = torch.matmul(inputs, ent_mat)
        value = inputs

        scores = torch.softmax(query @ key.transpose(-1, -2) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value


class FastBaseEstimator:
    """
    Classical estimator that evaluates a PyTorch model on a set of parameter
    vectors and aggregates the outputs with user supplied observables.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        if not observables:
            observables = [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    val = observable(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

    def evaluate_with_shots(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "ClassicalSelfAttention"]
