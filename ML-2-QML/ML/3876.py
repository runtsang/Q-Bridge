"""Hybrid classical estimator that mimics the quantum interface.

Features:
* Unified evaluate interface for torch models.
* Optional shot noise simulation to emulate quantum measurement statistics.
* Convenience factory for building classifier networks mirroring the quantum building block.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of scalar parameters into a 2‑D float tensor."""
    t = torch.as_tensor(values, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t


def build_classifier_circuit(
    num_features: int, depth: int
) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """Construct a simple feed‑forward classifier with metadata.

    The returned tuple mirrors the signature of the quantum counterpart:
    * ``model`` – the nn.Sequential object.
    * ``encoding`` – indices of input features that are directly encoded.
    * ``weight_sizes`` – number of trainable parameters per layer.
    * ``observables`` – placeholder indices for the output logits.
    """
    layers: list[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: list[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))  # indices of logits
    return network, encoding, weight_sizes, observables


class FastBaseEstimator:
    """Evaluate a torch model for a batch of parameter sets and observables.

    Parameters
    ----------
    model
        Any :class:`torch.nn.Module` instance that accepts a tensor of shape
        ``(batch, input_dim)`` and returns a tensor of shape ``(batch, output_dim)``.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a matrix of scalar values.

        Each row corresponds to a parameter set and each column to an observable.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for values in parameter_sets:
                inputs = _ensure_batch(values)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
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
        """Add Gaussian shot noise to the deterministic evaluations."""
        base = self.evaluate(observables, parameter_sets)
        if shots is None:
            return base
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in base:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "build_classifier_circuit"]
