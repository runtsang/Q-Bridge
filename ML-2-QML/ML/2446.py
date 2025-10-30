"""Hybrid classical estimator that unifies neural‑network evaluation and self‑attention primitives.

The estimator keeps the lightweight API of the original FastBaseEstimator while
adding:
* optional Gaussian shot noise (HybridEstimatorWithNoise)
* a built‑in ClassicalSelfAttention block that can be dropped into any nn.Module
* a convenience factory for a simple attention‑based network
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
ParameterSet = Sequence[float]
ParamSetSequence = Sequence[ParameterSet]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D parameter list to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class ClassicalSelfAttention(nn.Module):
    """A lightweight self‑attention block that mimics the quantum construction.

    The block accepts rotation and entanglement parameters that are reshaped
    into weight matrices.  It can be dropped into any torch.nn.Module.
    """
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        query = inputs @ rotation_params.reshape(self.embed_dim, -1)
        key   = inputs @ entangle_params.reshape(self.embed_dim, -1)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs

class HybridEstimator:
    """Evaluate a torch model or a self‑attention callable for many parameter sets.

    The estimator is deliberately lightweight: it runs the model in eval mode,
    collects the outputs and applies the supplied observables.  It is a drop‑in
    replacement for the original FastBaseEstimator with the same public API.
    """
    def __init__(self, model: Union[nn.Module, Callable[..., torch.Tensor]]) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: ParamSetSequence,
    ) -> List[List[float]]:
        """Compute observables for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output and returns a scalar
            (tensor or float).  If empty, the mean of the last dimension is used.
        parameter_sets : sequence of sequences
            Each inner sequence contains the parameters that will be fed to the
            model.  For a neural network the parameters are the input vector;
            for a self‑attention block they are the concatenated rotation and
            entanglement parameters followed by the input data.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        if isinstance(self.model, nn.Module):
            self.model.eval()
            with torch.no_grad():
                for params in parameter_sets:
                    batch = _ensure_batch(params)
                    outputs = self.model(batch)
                    row: List[float] = []
                    for obs in observables:
                        val = obs(outputs)
                        if isinstance(val, torch.Tensor):
                            scalar = float(val.mean().cpu())
                        else:
                            scalar = float(val)
                        row.append(scalar)
                    results.append(row)
        else:
            # Assume a callable that implements a forward‑like interface.
            for params in parameter_sets:
                batch = _ensure_batch(params)
                outputs = self.model(batch)
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

class HybridEstimatorWithNoise(HybridEstimator):
    """Same as HybridEstimator but injects Gaussian shot noise."""
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: ParamSetSequence,
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

def simple_attention_model(embed_dim: int = 4) -> nn.Module:
    """Convenience factory that returns a minimal network containing a
    ClassicalSelfAttention block followed by a linear read‑out.
    """
    return nn.Sequential(
        nn.Linear(embed_dim, embed_dim),
        ClassicalSelfAttention(embed_dim=embed_dim),
        nn.Linear(embed_dim, 1),
    )

__all__ = ["HybridEstimator", "HybridEstimatorWithNoise", "ClassicalSelfAttention", "simple_attention_model"]
