"""Hybrid classical classifier inspired by QML design."""

from __future__ import annotations

from typing import Iterable, Callable, List, Tuple, Sequence

import torch
import torch.nn as nn
import numpy as np

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def build_classifier_circuit(
    num_features: int,
    depth: int,
    hidden_dim: int | None = None,
) -> Tuple[nn.Module, Iterable[int], List[int], List[ScalarObservable]]:
    """
    Construct a feed‑forward neural classifier with metadata.

    Parameters
    ----------
    num_features : int
        Number of input features.
    depth : int
        Number of hidden layers.
    hidden_dim : int | None
        Width of hidden layers; defaults to ``num_features``.

    Returns
    -------
    network : nn.Module
        Sequential classifier.
    encoding : Iterable[int]
        Identity mapping of input features.
    weight_sizes : List[int]
        Parameter count per layer (weights + biases).
    observables : List[ScalarObservable]
        Default observable functions applied to the network output.
    """
    hidden_dim = hidden_dim or num_features
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, hidden_dim)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = hidden_dim

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)

    # Default observable: mean over the two output logits.
    observables: List[ScalarObservable] = [lambda out: out.mean(dim=-1)]

    return network, encoding, weight_sizes, observables


class QuantumClassifierModel(nn.Module):
    """
    Classical classifier mirroring the quantum helper interface.

    The class wraps a feed‑forward network and exposes an ``evaluate`` method
    compatible with the ``FastEstimator`` style.  Optional Gaussian shot
    noise emulates measurement statistics.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.network, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_features, depth, hidden_dim
        )
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def _ensure_batch(self, values: Sequence[float] | torch.Tensor) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            tensor = values
        else:
            tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Return observable values for each parameter set.

        Parameters
        ----------
        observables : Iterable[ScalarObservable] | None
            Functions applied to the network output.  If None, the
            default observable from ``build_classifier_circuit`` is used.
        parameter_sets : Sequence[Sequence[float]] | None
            Batched inputs to the classifier.  If None, an empty list is returned.
        shots : int | None
            Optional shot number for Gaussian noise injection.
        seed : int | None
            Random seed for reproducible noise.
        """
        if parameter_sets is None:
            return []

        obs = list(observables) or self.observables
        results: List[List[float]] = []

        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = self._ensure_batch(params)
                outputs = self(inputs)
                row: List[float] = []
                for observable in obs:
                    val = observable(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy_results: List[List[float]] = []
            for row in results:
                noisy_row = [
                    float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
                ]
                noisy_results.append(noisy_row)
            return noisy_results

        return results


__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
