"""Advanced hybrid estimator for classical neural networks with batch processing and GPU support."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Callable, Sequence

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Create a 2‑D tensor from a list of parameters, preserving batch dimension."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """Batch‑enabled estimator for PyTorch models with optional GPU acceleration and shot‑noise simulation."""

    def __init__(self, model: nn.Module, device: str | torch.device | None = None) -> None:
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

    def _prepare_batch(self, parameter_sets: Sequence[Sequence[float]]) -> torch.Tensor:
        batch = torch.as_tensor(parameter_sets, dtype=torch.float32, device=self.device)
        return batch

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate a batch of parameters against the model and observables."""
        if not observables:
            observables = [lambda out: out.mean(dim=-1)]
        observables = list(observables)
        batch = self._prepare_batch(parameter_sets)
        with torch.no_grad():
            outputs = self.model(batch)
        results: List[List[float]] = []
        for obs in observables:
            val = obs(outputs)
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            else:
                val = np.asarray(val)
            results.append(val)
        # Transpose results to match parameter set ordering
        return [list(row) for row in zip(*results)]

    def evaluate_with_noise(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Same as evaluate but injects Gaussian shot noise."""
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = [[float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                 for row in raw]
        return noisy

    def hybrid_loss(
        self,
        classical_outputs: List[float],
        quantum_outputs: List[complex],
        weight: float = 0.5,
    ) -> float:
        """Return weighted sum of classical and quantum outputs."""
        return weight * np.mean(classical_outputs) + (1 - weight) * np.mean(quantum_outputs)

__all__ = ["FastBaseEstimator"]
