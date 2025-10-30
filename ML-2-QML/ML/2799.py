"""Hybrid estimator module – classical implementation using PyTorch.

This module defines :class:`FastHybridEstimator` that evaluates a PyTorch
neural network on batches of parameters and returns scalar observables.
It also provides a convenience function
:meth:`build_classifier_circuit` mirroring the quantum variant, so users
can construct a compatible feed‑forward network and obtain metadata
(encoding, weight sizes, observables) in a single call.

The public API matches the original FastBaseEstimator while adding
shot‑noise support for stochastic experiments.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Turn a 1‑D list of parameters into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastHybridEstimator:
    """Evaluate a PyTorch model on batches of parameters.

    Parameters
    ----------
    model : nn.Module
        A neural network that accepts a batch of input features and
        produces a batch of outputs.

    Notes
    -----
    The estimator operates in evaluation mode and disables gradient
    computation.  It accepts an iterable of scalar observables that
    transform the network output into a scalar (or a list of scalars)
    per input.  If no observables are supplied, the mean of the last
    dimension of the network output is used.
    """

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
        """Compute observables for each parameter set.

        Parameters
        ----------
        observables
            Iterable of callables that map model outputs to scalars.
        parameter_sets
            Sequence of parameter sequences to evaluate.
        shots
            If provided, Gaussian shot noise with variance 1/shots is added
            to each observable value.  Useful for simulating measurement
            uncertainty.
        seed
            Seed for the random number generator used when `shots` is set.

        Returns
        -------
        List[List[float]]
            Nested list where each inner list contains the evaluated
            observables for a single parameter set.
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
) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Construct a feed‑forward classifier mirroring the quantum variant.

    Returns a tuple containing the model, the encoding indices,
    the weight sizes per layer, and the observable indices.
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


__all__ = ["FastHybridEstimator", "build_classifier_circuit"]
