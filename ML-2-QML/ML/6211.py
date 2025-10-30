from __future__ import annotations

import torch
from torch import nn
import numpy as np
from typing import Iterable, Callable, List, Sequence

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
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


class Conv:
    """Hybrid classical convolution filter with optional noise simulation.

    The class can be used as a drop‑in replacement for the quantum quanvolution
    in the original repository.  It exposes a ``run`` method that accepts a
    2‑D array and returns a scalar activation.  The ``evaluate`` method
    accepts a list of parameter sets and a list of scalar observables,
    mirroring the API of the quantum FastBaseEstimator.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, device: str = "cpu") -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.device = device
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True).to(device)

    def run(self, data) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate the filter for a batch of weight sets.

        Parameters
        ----------
        observables : Iterable[Callable[[Tensor], Tensor | float]]
            Functions that map the convolution output to a scalar.
        parameter_sets : Sequence[Sequence[float]]
            Each inner sequence contains ``kernel_size**2`` weights
            that replace the filter’s parameters.
        shots : int | None
            If provided, Gaussian noise with variance 1/shots is added.
        seed : int | None
            Seed for the noise generator.
        """
        estimator = FastEstimator(self)
        results = estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)
        return results


__all__ = ["Conv", "FastBaseEstimator", "FastEstimator"]
