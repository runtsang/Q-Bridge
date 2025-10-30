from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor | np.ndarray], torch.Tensor | np.ndarray | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridBaseEstimator:
    """Hybrid estimator that can wrap a PyTorch model or a self‑attention block."""

    def __init__(self, model: Union[nn.Module, object]) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate the wrapped model for each parameter set.

        Args:
            observables: Callables that map the model output to a scalar.
            parameter_sets: Sequence of parameter vectors.
            shots: Optional number of shots for stochastic estimation.
            seed: Optional RNG seed for reproducibility.

        Returns:
            A list of rows, each containing the scalar values for the
            provided observables.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        if isinstance(self.model, nn.Module):
            self.model.eval()
            with torch.no_grad():
                for params in parameter_sets:
                    inputs = _ensure_batch(params)
                    outputs = self.model(inputs)
                    row: List[float] = []
                    for observable in observables:
                        val = observable(outputs)
                        if isinstance(val, torch.Tensor):
                            scalar = float(val.mean().cpu())
                        else:
                            scalar = float(val)
                        row.append(scalar)
                    results.append(row)
        else:
            # Assume a self‑attention or other callable with a `run` method.
            for params in parameter_sets:
                out = self.model.run(*params)
                row: List[float] = []
                for observable in observables:
                    val = observable(out)
                    if isinstance(val, np.ndarray):
                        scalar = float(np.mean(val))
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy

        return results


__all__ = ["HybridBaseEstimator"]
