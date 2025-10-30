"""Extended FastBaseEstimator for PyTorch with gradient support and optional GPU usage."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List

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
    """Evaluate neural networks for batches of inputs and observables with optional GPU support.

    The class retains the lightweight API of the original seed but adds:
    * Automatic gradient computation of observables w.r.t. network parameters.
    * Optional device selection (CPU or CUDA).
    * A ``compute_gradients`` method that returns NumPy arrays of gradients.
    * A ``evaluate`` method that can add Gaussian shot noise.
    """

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Deterministic evaluation of observables for each parameter set."""
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
                    scalar = float(value.mean().cpu()) if isinstance(value, torch.Tensor) else float(value)
                    row.append(scalar)
                results.append(row)
        return results

    def compute_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[np.ndarray]]:
        """Return gradients of each observable w.r.t. the network parameters."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        gradients: List[List[np.ndarray]] = []
        self.model.eval()
        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device)
            inputs.requires_grad_(True)
            outputs = self.model(inputs)
            row: List[np.ndarray] = []
            for observable in observables:
                value = observable(outputs)
                scalar = value.mean() if isinstance(value, torch.Tensor) else torch.tensor(value, device=self.device)
                grads = torch.autograd.grad(
                    scalar,
                    inputs,
                    retain_graph=True,
                    create_graph=False,
                )[0]
                row.append(grads.cpu().numpy())
            gradients.append(row)
        return gradients

    def evaluate_with_noise(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Wrap ``evaluate`` and add Gaussian shot noise if ``shots`` is provided."""
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
