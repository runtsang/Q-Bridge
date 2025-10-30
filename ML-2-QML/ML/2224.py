"""Hybrid estimator module: Classical GNN based estimator.

This module defines UnifiedGraphEstimator that uses a lightweight feed‑forward
graph‑neural‑network to map input features to scalar predictions.  It
supports batched evaluation and optional Gaussian shot‑noise simulation,
mirroring the FastEstimator interface from the seed project but with a
fully classical implementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class ClassicalGNN(nn.Module):
    """Feed‑forward GNN that emulates the architecture of the seed GraphQNN."""

    def __init__(self, arch: Sequence[int]) -> None:
        super().__init__()
        self.arch = arch
        self.layers = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(arch[:-1], arch[1:])]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = torch.tanh(layer(out))
        return out


class UnifiedGraphEstimator:
    """Classical estimator that maps inputs to outputs via a GNN."""

    def __init__(self, gnn_arch: Sequence[int], *, shots: int | None = None, seed: int | None = None) -> None:
        self.gnn = ClassicalGNN(gnn_arch)
        self.shots = shots
        self.seed = seed

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate a batch of inputs through the GNN and apply observables.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the GNN output tensor and returns a scalar
            or a tensor that will be reduced to a scalar.
        parameter_sets : sequence of sequences
            Each inner sequence represents an input feature vector.

        Returns
        -------
        List[List[float]]
            A list where each element corresponds to a parameter set and
            contains the evaluation of all observables.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.gnn.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.gnn(inputs)
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)

        if self.shots is None:
            return results

        rng = np.random.default_rng(self.seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["UnifiedGraphEstimator"]
