"""Hybrid quantum‑classical classifier factory and estimator.

Provides a classical neural network with the same API as the quantum
variant, enabling side‑by‑side experiments.  The model exposes:
* `build_classifier_circuit(num_features, depth)` – returns a
  Sequential, encoding indices, weight counts and observable indices.
* `FastEstimator` – evaluates the network with optional Gaussian
  shot noise, mirroring the quantum estimator.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Callable

import torch
import torch.nn as nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Iterable[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastEstimator:
    """Deterministic estimator with optional Gaussian shot noise."""

    def __init__(self, model: nn.Module) -> None:
        self._model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None,
        parameter_sets: Iterable[Iterable[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        if observables is None:
            observables = [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self._model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self._model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)

        if shots is None:
            return results

        import numpy as np
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


def build_classifier_circuit(
    num_features: int, depth: int
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Return a feed‑forward classifier and metadata.

    Parameters
    ----------
    num_features: int
        Input dimensionality.  The network will have *depth* hidden
        layers each of size *num_features*.
    depth: int
        Number of hidden layers.

    Returns
    -------
    model: nn.Module
        Sequential network.
    encoding: list[int]
        Indices of input features used for encoding (identity mapping).
    weight_sizes: list[int]
        Number of trainable parameters per layer.
    observables: list[int]
        Dummy observable indices used by the quantum interface.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        lin = nn.Linear(in_dim, num_features)
        layers.append(lin)
        layers.append(nn.ReLU())
        weight_sizes.append(lin.weight.numel() + lin.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    model = nn.Sequential(*layers)
    observables = list(range(2))
    return model, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit", "FastEstimator"]
