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

class HybridFastEstimator:
    """Fast batched estimator for classical neural nets with optional shot noise.

    Parameters
    ----------
    model : nn.Module
        A PyTorch model that accepts a batch of inputs and returns a batch
        of outputs.
    shots : int | None, optional
        If provided, Gaussian shot noise is added to the deterministic
        outputs.  The noise standard deviation is ``max(1e-6, 1 / shots)``.
    seed : int | None, optional
        Seed for the random number generator used to generate shot noise.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.model = model
        self.shots = shots
        self.seed = seed

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
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

        if self.shots is None:
            return results

        rng = np.random.default_rng(self.seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__: list[str] = ["HybridFastEstimator"]
