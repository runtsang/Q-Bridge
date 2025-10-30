"""FastBaseEstimatorGen301: Classical neural‑network evaluator with optional shot noise."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimatorGen301:
    """Evaluate a PyTorch model for many parameter sets.

    Parameters
    ----------
    model : nn.Module
        Neural network to evaluate. Must accept a batch of 1‑D tensors.
    device : str | torch.device, optional
        Device on which to run the model. Defaults to ``'cpu'``.
    """

    def __init__(self, model: nn.Module, device: torch.device | str = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Compute observables for each parameter set.

        Parameters
        ----------
        observables : iterable of callables, optional
            Each callable receives the model output and returns a scalar
            value or a tensor that can be reduced to a scalar. If omitted a
            single default observable that averages over the last dimension
            is used.
        parameter_sets : sequence of sequence of floats, optional
            Each inner sequence is a set of parameters fed to the model.
        shots : int, optional
            If provided, Gaussian noise with variance 1/shots is added
            to each mean value to emulate shot noise.
        seed : int, optional
            Random seed for the noise generator.

        Returns
        -------
        List[List[float]]
            A list of rows, one per parameter set, each row containing
            the observable values.
        """
        if parameter_sets is None:
            return []

        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
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

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimatorGen301"]
