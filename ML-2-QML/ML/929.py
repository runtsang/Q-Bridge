"""Lightweight estimator utilities implemented with PyTorch modules.

This module extends the original FastBaseEstimator by adding:
* ``FastEstimator`` – optional Gaussian shot noise.
* ``HybridEstimator`` – combines a PyTorch model with a callable quantum output.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a list of scalars into a batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for a list of parameter sets.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.
    """
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


class HybridEstimator:
    """Combine a classical PyTorch model and a quantum output callable.

    The quantum part is supplied as a callable that accepts a sequence of
    parameters and returns a torch.Tensor of shape (output_dim,).  The
    outputs of the two parts are fused (concatenated or added) and
    optionally passed through a final linear layer.

    Parameters
    ----------
    classical : nn.Module
        Classical model that maps the classical parameters to a tensor.
    quantum : Callable[[Sequence[float]], torch.Tensor]
        Callable returning a quantum output tensor.
    fusion : str
        ``'concat'`` (default) or ``'add'`` to fuse the two outputs.
    fusion_layer : nn.Module | None
        Optional linear layer applied after fusion.
    """
    def __init__(
        self,
        classical: nn.Module,
        quantum: Callable[[Sequence[float]], torch.Tensor],
        *,
        fusion: str = "concat",
        fusion_layer: nn.Module | None = None,
    ) -> None:
        if fusion not in {"concat", "add"}:
            raise ValueError("fusion must be 'concat' or 'add'")
        self.classical = classical
        self.quantum = quantum
        self.fusion = fusion
        self.fusion_layer = fusion_layer

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate the hybrid model for all parameter sets."""
        results: List[List[float]] = []
        self.classical.eval()
        with torch.no_grad():
            for params in parameter_sets:
                # Classical part
                cls_out = self.classical(_ensure_batch(params))
                # Quantum part
                q_out = self.quantum(params)  # Expected shape (output_dim,)
                if self.fusion == "concat":
                    fused = torch.cat([cls_out, q_out], dim=-1)
                else:  # add
                    fused = cls_out + q_out
                if self.fusion_layer is not None:
                    fused = self.fusion_layer(fused)
                # Compute observables on fused tensor
                row: List[float] = []
                for obs in observables:
                    val = obs(fused)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results


__all__ = ["FastBaseEstimator", "FastEstimator", "HybridEstimator"]
