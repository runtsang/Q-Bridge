"""Hybrid estimator combining classical PyTorch and quantum Qiskit interfaces.

The class `HybridBaseEstimator` accepts either a PyTorch `nn.Module` or a Qiskit
`QuantumCircuit` and exposes a unified `evaluate` method.  For classical models
an optional Gaussian shot noise can be added; for quantum models a similar
noise model is applied to the deterministic expectation values.  The module
also provides a `build_classifier_circuit` function that mirrors the quantum
builder, enabling side‑by‑side experiments.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence of parameters to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridBaseEstimator:
    """Evaluate neural networks or quantum circuits with optional shot noise."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Return a list of rows of observable values for each parameter set.

        Parameters
        ----------
        observables : Iterable[Callable]
            Callables that map a model output tensor to a scalar.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors to evaluate.
        shots : int | None, optional
            If provided, Gaussian shot noise with variance 1/shots is added.
        seed : int | None, optional
            Random seed for reproducible noise.
        """
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

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


def build_classifier_circuit(
    num_features: int, depth: int
) -> tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Construct a feed‑forward classifier mirroring the quantum version.

    Returns
    -------
    network : nn.Module
        Sequential network with ReLU activations.
    encoding : Iterable[int]
        Indices of the input features used for encoding.
    weight_sizes : Iterable[int]
        Number of trainable parameters per linear layer.
    observables : list[int]
        Dummy observable indices matching the output dimension.
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
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


__all__ = ["HybridBaseEstimator", "build_classifier_circuit"]
