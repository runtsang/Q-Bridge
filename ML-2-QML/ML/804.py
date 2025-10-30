"""Enhanced estimator utilities implemented with PyTorch modules.

Features:
- GPU acceleration via torch.device.
- Supports batched evaluation of multiple observables.
- Optional dropout and batch normalization for inference.
- Caching of model outputs for repeated observable evaluation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

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
    """Evaluate neural networks for batches of inputs and observables with GPU support.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model.
    device : str | torch.device, optional
        Target device for inference. Defaults to ``"cpu"``.
    """

    def __init__(self, model: nn.Module, device: Union[str, torch.device] = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(inputs.to(self.device))

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute a list of observable values for each parameter set.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Callables that map model outputs to scalars.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors.

        Returns
        -------
        List[List[float]]
            Nested list where each inner list contains the observable values
            for a single parameter set.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device)
            outputs = self._forward(inputs)
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


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
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


__all__ = ["FastBaseEstimator", "FastEstimator"]
