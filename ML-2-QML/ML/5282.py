"""Hybrid classical classifier utilities with fast batched evaluation.

The module defines :class:`SharedClassifier` which can build a
feed‑forward network and evaluate it efficiently.  It also exposes a
small regression network inspired by the EstimatorQNN example and
provides a noisy estimator that mimics quantum shot noise.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class SharedClassifier:
    """Factory and evaluator for hybrid classical models."""

    @staticmethod
    def build_classifier(
        num_features: int, depth: int
    ) -> nn.Module:
        """
        Return a linear classifier with ``depth`` hidden layers.

        Each hidden layer has ``num_features`` units and a ReLU
        activation.  The final layer maps to a two‑class output.
        We clip the weight matrices to ``[-5, 5]`` to keep the
        optimisation stable, mirroring the photonic fraud detection
        implementation.
        """
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            with torch.no_grad():
                linear.weight.clamp_(-5.0, 5.0)
                linear.bias.clamp_(-5.0, 5.0)
            layers.extend([linear, nn.ReLU()])
            in_dim = num_features
        head = nn.Linear(in_dim, 2)
        with torch.no_grad():
            head.weight.clamp_(-5.0, 5.0)
            head.bias.clamp_(-5.0, 5.0)
        layers.append(head)
        return nn.Sequential(*layers)

    @staticmethod
    def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def evaluate(
        model: nn.Module,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Evaluate ``model`` for a batch of inputs.

        Each observable is a callable that maps the network output
        to a scalar.  The function is a lightweight re‑implementation
        of :class:`FastBaseEstimator` from the reference.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = SharedClassifier._ensure_batch(params)
                outputs = model(inputs)
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results

    @staticmethod
    def evaluate_with_shots(
        model: nn.Module,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Add Gaussian shot‑noise to the deterministic evaluation.
        """
        raw = SharedClassifier.evaluate(model, observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy

    @staticmethod
    def regression_network() -> nn.Module:
        """
        Return a small regression network similar to EstimatorQNN.
        """
        return nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )


__all__ = ["SharedClassifier"]
