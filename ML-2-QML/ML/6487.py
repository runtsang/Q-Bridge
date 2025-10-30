"""Hybrid estimator implemented in PyTorch."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridEstimator:
    """Evaluate a PyTorch model for a batch of parameter sets.

    Parameters
    ----------
    model : nn.Module
        Any PyTorch network that maps a batch of floats to a tensor of
        outputs.  The network can be a simple feed‑forward net or a
        recurrent layer.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    def _apply_observables(
        self,
        outputs: torch.Tensor,
        observables: Iterable[ScalarObservable],
    ) -> List[float]:
        """Apply a list of scalar observables to the network output."""
        results: List[float] = []
        for obs in observables:
            val = obs(outputs)
            if isinstance(val, torch.Tensor):
                val = float(val.mean().cpu())
            else:
                val = float(val)
            results.append(val)
        return results

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Run the network on each parameter set and compute all observables.

        When *shots* is provided a Gaussian shot‑noise model is added to
        the deterministic outputs.  This mimics a noisy quantum device
        while keeping the evaluation purely classical.
        """
        raw: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                batch = _ensure_batch(params)
                out = self.model(batch)
                raw.append(self._apply_observables(out, observables))

        if shots is None:
            return raw

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy.append(
                [rng.normal(mean, max(1e-6, 1 / shots)) for mean in row]
            )
        return noisy


__all__ = ["HybridEstimator"]
