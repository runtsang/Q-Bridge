"""Hybrid quantum‑classical classifier – classical module."""

from __future__ import annotations

from typing import Iterable, Tuple, List, Callable
import torch
import torch.nn as nn
import numpy as np

# --------------------------------------------------------------------------- #
#  Classical estimator utilities (adapted from FastEstimator.py)
# --------------------------------------------------------------------------- #
class _FastBaseEstimator:
    """Deterministic evaluator for a torch model."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Iterable[Iterable[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    row.append(float(val.mean().cpu()) if isinstance(val, torch.Tensor) else float(val))
                results.append(row)
        return results

class FastEstimator(_FastBaseEstimator):
    """Evaluator with optional Gaussian shot‑noise."""
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
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
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

# --------------------------------------------------------------------------- #
#  Classical circuit builder (mirrors the quantum version)
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a feed‑forward classifier with the same depth and feature size as
    the quantum ansatz.  The returned metadata (encoding, weight_sizes, observables)
    match the quantum function to ease hybrid experiments.
    """
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

    net = nn.Sequential(*layers)
    observables = list(range(2))
    return net, encoding, weight_sizes, observables

# --------------------------------------------------------------------------- #
#  Public hybrid class
# --------------------------------------------------------------------------- #
class QuantumClassifierModel:
    """
    Dual‑mode classifier that can be instantiated with either a torch model
    or a Qiskit QuantumCircuit.  In classical mode it uses FastEstimator;
    in quantum mode it uses the quantum FastBaseEstimator implementation
    (imported lazily to avoid heavy dependencies when only the classical
    side is needed).
    """
    def __init__(self, model: nn.Module | "QuantumCircuit"):
        self._model = model
        if isinstance(model, nn.Module):
            self.estimator = FastEstimator(model)
        else:
            # Lazy import – only executed when a quantum circuit is supplied.
            from.FastBaseEstimator import FastBaseEstimator as _QFastBaseEstimator
            self.estimator = _QFastBaseEstimator(model)

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]] | Iterable["BaseOperator"],
        parameter_sets: Iterable[Iterable[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        return self.estimator.evaluate(
            observables, parameter_sets, shots=shots, seed=seed
        )

__all__ = ["build_classifier_circuit", "FastEstimator", "QuantumClassifierModel"]
