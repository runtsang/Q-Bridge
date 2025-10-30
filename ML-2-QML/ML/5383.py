"""Hybrid classical binary classifier with graph-based feature extraction and optional shot-noise evaluation.

This module merges concepts from the original hybrid CNN, the FastEstimator utilities, a
graph neural network inspired by GraphQNN, and a regression-style dataset generator.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Dataset utilities ----------
def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a toy binary classification dataset by mapping a superposition angle to a label."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    probs = 0.5 * (1 + np.sin(angles))
    y = (probs > 0.5).astype(np.float32)
    return x, y

class ClassificationDataset(torch.utils.data.Dataset):
    """Simple PyTorch dataset wrapping the superposition data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# ---------- GraphQNN utilities ----------
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_graph_network(qnn_arch: Sequence[int], samples: int) -> Tuple[Sequence[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Generate a random network architecture and training data."""
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = []
    for _ in range(samples):
        features = torch.randn(target_weight.size(1))
        target = target_weight @ features
        training_data.append((features, target))
    return qnn_arch, weights, training_data, target_weight

def feedforward(qnn_arch: Sequence[int], weights: Sequence[torch.Tensor], samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
    """Forward propagate a batch through a purely linear graph network."""
    outputs: List[List[torch.Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        outputs.append(activations)
    return outputs

# ---------- FastEstimator utilities ----------
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """Deterministic evaluator for a PyTorch model."""
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
    """Estimator with optional Gaussian shot noise."""
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

# ---------- Hybrid classifier ----------
class HybridBinaryClassifier(nn.Module):
    """
    A hybrid binary classifier that first maps inputs through a lightweight
    graph neural network (inspired by GraphQNN) and then applies a classical
    linear head.  The model can be wrapped by FastEstimator for noisy evaluation.
    """
    def __init__(self, num_features: int, qnn_arch: Sequence[int]) -> None:
        super().__init__()
        self.graph_arch = qnn_arch
        self.graph_weights = nn.ParameterList(
            [nn.Parameter(_random_linear(in_f, out_f)) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
        )
        self.head = nn.Linear(qnn_arch[-1], 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, features)
        activations = inputs
        for weight in self.graph_weights:
            activations = torch.tanh(weight @ activations.t()).t()
        logits = self.head(activations)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        self.eval()
        with torch.no_grad():
            return self.forward(inputs)

    def evaluate(self, inputs: torch.Tensor, *, shots: int | None = None, seed: int | None = None) -> List[float]:
        """
        Evaluate the model on a batch of inputs, optionally adding shot noise.
        """
        estimator_cls = FastEstimator if shots else FastBaseEstimator
        estimator = estimator_cls(self)
        observables = [lambda out: out[:,0]]
        parameter_sets = [input.tolist() for input in inputs]
        results = estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)
        return [row[0] for row in results]

__all__ = [
    "HybridBinaryClassifier",
    "FastBaseEstimator",
    "FastEstimator",
    "ClassificationDataset",
    "generate_superposition_data",
]
