"""UnifiedFastEstimator combines classical PyTorch inference with optional shot‑noise simulation.

The class exposes a lightweight interface that can evaluate a PyTorch
neural network on batches of input parameters, compute scalar observables,
and optionally add Gaussian shot noise to emulate finite‑sample effects.
It also includes helper functions for generating synthetic regression data
and a simple Torch Dataset.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of floats to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class UnifiedFastEstimator:
    """Evaluate a PyTorch model for a batch of parameters.

    Parameters
    ----------
    model : nn.Module
        A PyTorch neural network that takes a batch tensor of shape
        (batch_size, num_features) and returns a tensor of shape
        (batch_size, output_dim).
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """Compute observables for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output tensor and returns
            either a scalar tensor or a Python float.
        parameter_sets : sequence of sequences
            Each inner sequence contains the input parameters for one
            evaluation.  The inner dimension must match the model's
            input size.
        shots : int, optional
            If provided, Gaussian noise with variance 1/shots is added
            to each deterministic result to mimic shot noise.
        seed : int, optional
            Random seed for reproducible noise generation.

        Returns
        -------
        List[List[float]]
            Outer list corresponds to the parameter sets; inner list
            contains the value of each observable.
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
        noisy_results: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy_results.append(noisy_row)
        return noisy_results


def generate_classical_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data for a classical model.

    The data is sampled uniformly in [-1, 1]^num_features.  The target
    is a smooth function of the sum of the features:
        y = sin(sum(x)) + 0.1 * cos(2 * sum(x))
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that returns states and targets for regression."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_classical_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


__all__ = ["UnifiedFastEstimator", "generate_classical_data", "RegressionDataset"]
