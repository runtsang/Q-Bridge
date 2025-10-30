"""Enhanced estimator for neural network models with batched evaluation,
GPU acceleration, and optional shot noise simulation.

The class `FastBaseEstimatorGen128` extends the original lightweight
estimator by supporting vectorized observables, GPU inference, and
a flexible noise model that can be toggled during evaluation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of floats to a 2‑D torch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimatorGen128:
    """Evaluate neural networks for batches of inputs and observables.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.
    device : str | torch.device, optional
        Target device for inference. Defaults to ``cpu``.
    """

    def __init__(self, model: nn.Module, device: torch.device | str = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
    ) -> List[List[float]]:
        """Compute observables for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output and returns a scalar
            or a tensor that can be reduced to a scalar.
        parameter_sets : sequence of sequences
            Each inner sequence contains the parameters for one forward pass.
        Returns
        -------
        List[List[float]]
            Nested list with one row per parameter set and one column per
            observable.
        """
        if observables is None:
            observables = [lambda outputs: outputs.mean(dim=-1)]
        if parameter_sets is None:
            return []

        self.model.eval()
        results: List[List[float]] = []

        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
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

    def evaluate_with_shots(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Same as :meth:`evaluate` but adds Gaussian shot noise.

        Parameters
        ----------
        shots : int, optional
            Number of shots to simulate. If ``None`` no noise is added.
        seed : int, optional
            Random seed for reproducibility.
        """
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [
                rng.normal(mean, max(1e-6, 1 / shots)) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimatorGen128"]
