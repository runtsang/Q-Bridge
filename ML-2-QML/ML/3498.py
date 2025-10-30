"""Hybrid estimator combining PyTorch model evaluation with optional shot noise and flexible observables."""
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


class FastBaseEstimator:
    """Evaluate a PyTorch model for batches of inputs and a list of observables."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Optional[Iterable[ScalarObservable]] = None,
        parameter_sets: Sequence[Sequence[float]] = (),
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Parameters
        ----------
        observables : Optional[Iterable[ScalarObservable]]
            Callables that turn the model output into a scalar.  If ``None`` a
            default mean over the last dimension is used.
        parameter_sets : Sequence[Sequence[float]]
            A list of parameter vectors that will be fed to the model.
        shots : Optional[int]
            When supplied, Gaussian shot noise with variance 1/shots is added
            to the deterministic result.
        seed : Optional[int]
            Random seed for reproducible shot noise.
        """
        observables = list(observables or [lambda out: out.mean(dim=-1)])
        results: List[List[float]] = []

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

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy

        return results


def FCL(n_features: int = 1) -> nn.Module:
    """
    A classical stand‑in for a fully connected quantum layer.

    Returns an nn.Module with a ``run`` method mimicking the quantum example.
    The module applies a single linear layer followed by a tanh non‑linearity
    and returns the mean over the batch.
    """
    class FullyConnectedLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().cpu().numpy()

    return FullyConnectedLayer()


__all__ = ["FastBaseEstimator", "FCL"]
