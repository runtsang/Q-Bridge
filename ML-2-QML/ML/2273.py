"""Hybrid fast estimator for PyTorch models with optional shot noise.

This module extends the original FastBaseEstimator by adding
support for:
- Batch evaluation of multiple parameter sets.
- Optional Gaussian shot noise to mimic measurement statistics.
- A classifier factory mirroring the quantum interface.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of parameters to a 2‑D tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for a list of parameter sets.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.  The network must accept a
        batch of parameters of shape ``(batch, in_features)``.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Return a matrix of observable values.

        Each row corresponds to a parameter set in
        ``parameter_sets`` and each column to an observable in
        ``observables``.  If ``shots`` is supplied, Gaussian noise with
        variance ``1/shots`` is added to each mean value to emulate
        finite‑shot statistics.
        """
        if observables is None:
            observables = [lambda outputs: outputs.mean(dim=-1)]
        if parameter_sets is None:
            return []

        observables = list(observables)
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

        # add shot noise if requested
        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [
                    float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
                ]
                noisy.append(noisy_row)
            results = noisy

        return results


# --------------------------------------------------------------------------- #
#  Classifier factory – classical analogue of the quantum circuit builder
# --------------------------------------------------------------------------- #
def build_classifier_circuit(
    num_features: int, depth: int
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Return a feed‑forward network and metadata.

    The returned tuple mimics the signature of the quantum helper:
    ``(model, encoding, weight_sizes, observables)``.
    """
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []

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


__all__ = ["FastBaseEstimator", "build_classifier_circuit"]
