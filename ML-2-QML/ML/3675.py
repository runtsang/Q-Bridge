"""Hybrid estimator for classical neural networks with optional shot noise.

The class is a drop‑in replacement for FastBaseEstimator / FastEstimator
but accepts any torch.nn.Module.  It also exposes a convenience factory
for the tiny EstimatorQNN network used in the reference seeds.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a list of scalars into a batched 2‑D tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridFastEstimator:
    """Evaluate a torch neural net for a batch of inputs and observables.

    Parameters
    ----------
    model : nn.Module
        Any neural network that accepts a 2‑D tensor of shape (batch, features)
        and returns a 2‑D tensor of shape (batch, outputs).

    Notes
    -----
    The estimator is intentionally lightweight: it disables gradients and
    uses ``torch.no_grad()``.  It mimics the original FastBaseEstimator
    but adds a static factory ``default_model`` that builds the
    EstimatorQNN architecture from the second reference pair.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Return the value of each observable for each parameter set.

        Parameters
        ----------
        observables
            Callables that map a model output tensor to a scalar.  If omitted
            a single mean‑over‑features observable is used.
        parameter_sets
            Iterable of parameter vectors that will be fed to the model.
        shots
            If provided, Gaussian noise with variance ``1/shots`` is added
            to each deterministic mean.
        seed
            RNG seed for reproducibility of the shot noise.

        Returns
        -------
        List[List[float]]
            A matrix of shape (len(parameter_sets), len(observables)).
        """
        if parameter_sets is None:
            return []

        observables = list(observables or [lambda outputs: outputs.mean(dim=-1)])
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    scalar = float(value.mean().cpu()) if isinstance(value, torch.Tensor) else float(value)
                    row.append(scalar)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    @staticmethod
    def default_model() -> nn.Module:
        """Return the tiny EstimatorQNN network (2‑→8‑→4‑→1)."""
        class EstimatorQNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(2, 8),
                    nn.Tanh(),
                    nn.Linear(8, 4),
                    nn.Tanh(),
                    nn.Linear(4, 1),
                )

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                return self.net(inputs)

        return EstimatorQNN()


__all__ = ["HybridFastEstimator"]
