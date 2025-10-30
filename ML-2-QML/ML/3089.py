from __future__ import annotations

import torch
import torch.nn as nn
from typing import Iterable, Tuple, List, Callable
import numpy as np

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Iterable[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridQuantumEstimator:
    """Base estimator that evaluates a PyTorch model on a batch of parameters."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Iterable[Iterable[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
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


class HybridQuantumNoisyEstimator(HybridQuantumEstimator):
    """Adds Gaussian shot noise to emulate quantum sampling."""
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Iterable[Iterable[float]],
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
            noisy.append([float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row])
        return noisy


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Constructs a feed‑forward neural network that mirrors the quantum ansatz."""
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


__all__ = [
    "HybridQuantumEstimator",
    "HybridQuantumNoisyEstimator",
    "build_classifier_circuit",
]
