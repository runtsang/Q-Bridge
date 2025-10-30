"""FastBaseEstimator for classical PyTorch models with shot‑noise support."""

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


class FastBaseEstimator:
    """Evaluate a PyTorch model on batches of parameters and observables.

    Parameters
    ----------
    model : nn.Module
        A forward‑only PyTorch model that accepts a batch of inputs.
    device : str | torch.device | None, optional
        Target device for inference.  ``None`` defaults to ``torch.device('cpu')``.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device | str] = None) -> None:
        self.model = model
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute scalar outputs for each observable.

        Parameters
        ----------
        observables : Iterable[Callable[[Tensor], Tensor | float]]
            Callables that map the model output to a scalar.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors to feed to the model.

        Returns
        -------
        List[List[float]]
            A 2‑D list where rows correspond to parameter sets and columns to observables.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
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

    def evaluate_noisy(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Wrap :meth:`evaluate` with Gaussian shot‑noise.

        Parameters
        ----------
        shots : int | None
            Number of simulated shots; ``None`` disables noise.
        seed : int | None
            Random seed for reproducibility.
        """
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator"]
