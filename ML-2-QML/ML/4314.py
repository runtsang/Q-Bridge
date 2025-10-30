from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
ModelType = Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence of floats into a 2‑D torch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastHybridEstimator:
    """
    Hybrid estimator that unifies the lightweight batch evaluation of
    FastBaseEstimator with the optional shot‑noise modelling of FastEstimator
    and the functional flexibility of a stand‑in quantum layer.

    It accepts either a PyTorch ``nn.Module`` or any callable that maps a
    ``torch.Tensor`` to a ``torch.Tensor``.  Observables are callables that
    transform the model output into a scalar.  An optional Gaussian noise
    term can be added to emulate finite‑shot sampling.
    """

    def __init__(self, model: ModelType) -> None:
        self.model = model

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs) if isinstance(self.model, nn.Module) else self.model(inputs)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate the model for each parameter set and observable.

        Parameters
        ----------
        observables:
            Sequence of callables mapping model outputs to scalars.
        parameter_sets:
            Iterable of parameter sequences to feed to the model.
        shots:
            If provided, Gaussian noise with variance 1/shots is added
            to each observable value to emulate finite‑shot sampling.
        seed:
            Random seed for reproducibility of shot noise.
        """
        observables = list(observables) or [lambda x: x.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self._forward(inputs)
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


__all__ = ["FastHybridEstimator"]
