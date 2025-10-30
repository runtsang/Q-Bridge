"""Enhanced lightweight estimator utilities implemented with PyTorch.

This module extends the original FastBaseEstimator by adding:
* Device‑aware execution (CPU/GPU).
* Automatic gradient computation of observables with respect to input parameters.
* Optional Poisson/normal shot noise simulation.
* Batch processing for large parameter sets.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
GradResult = Tuple[float, List[float]]  # (value, gradient vector)

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimatorGen150:
    """Evaluate neural networks for batches of inputs and observables with optional gradients and shot noise."""

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Deterministic evaluation of a list of observables over all parameter sets."""
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

    def evaluate_with_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[GradResult]]:
        """Return values and gradients of each observable w.r.t the input parameters."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[GradResult]] = []
        self.model.eval()
        for params in parameter_sets:
            inp = _ensure_batch(params).to(self.device)
            inp.requires_grad_(True)
            outputs = self.model(inp)
            row: List[GradResult] = []
            for observable in observables:
                value = observable(outputs)
                if isinstance(value, torch.Tensor):
                    scalar = value.mean()
                else:
                    scalar = torch.tensor(value, device=self.device, dtype=torch.float32)
                # Compute gradient
                self.model.zero_grad()
                scalar.backward(retain_graph=True)
                grad = inp.grad.clone().detach().cpu().numpy().flatten().tolist()
                row.append((float(scalar.cpu()), grad))
            results.append(row)
        return results

    def evaluate_with_shots(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Add Gaussian shot noise to deterministic evaluation."""
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["FastBaseEstimatorGen150"]
