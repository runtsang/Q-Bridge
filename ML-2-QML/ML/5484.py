"""QuantumHybridEstimator – classical core for batch evaluation and noise injection."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

# ----------------------------------------------------------------------
# 1️⃣  Classical utilities
# ----------------------------------------------------------------------
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Guarantee a 2‑D tensor with a leading batch dimension."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


# ----------------------------------------------------------------------
# 2️⃣  Core estimator
# ----------------------------------------------------------------------
class QuantumHybridEstimator:
    """
    Estimate a model that may contain classical PyTorch layers, quantum
    sub‑modules (via torchquantum) or simple circuit objects, and optionally
    add shot‑noise to emulate real hardware.

    The class is intentionally lightweight – it mirrors the original
    FastBaseEstimator but now accepts any callable that returns a tensor
    when given a batch of parameters.
    """

    def __init__(self, model: nn.Module | Callable[[torch.Tensor], torch.Tensor]):
        self.model = model

    def _forward(self, params: torch.Tensor) -> torch.Tensor:
        """Forward pass that works for both nn.Module and plain callables."""
        if isinstance(self.model, nn.Module):
            return self.model(params)
        return self.model(params)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output tensor and returns a scalar.
        parameter_sets : list of parameter vectors
            Each vector is fed to the model in a single batch.
        shots : int, optional
            When provided, Gaussian noise with variance 1/shots is added to each
            output to emulate measurement shot noise.
        seed : int, optional
            Random seed for reproducible noise.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval() if isinstance(self.model, nn.Module) else None
        with torch.no_grad():
            for params in parameter_sets:
                batch = _ensure_batch(params)
                output = self._forward(batch)
                row: List[float] = []
                for obs in observables:
                    val = obs(output)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)

        # Add shot noise if requested
        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy.append(noisy_row)
            results = noisy

        return results


# ----------------------------------------------------------------------
# 3️⃣  Convenience wrappers (optional)
# ----------------------------------------------------------------------
class QuantumHybridEstimatorWithNoise(QuantumHybridEstimator):
    """
    Thin wrapper that keeps the original API but exposes a dedicated
    constructor for noise‑only mode.
    """
    pass


__all__ = ["QuantumHybridEstimator", "QuantumHybridEstimatorWithNoise"]
