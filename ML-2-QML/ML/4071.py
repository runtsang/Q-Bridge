from __future__ import annotations

from typing import Iterable, Sequence, Callable, List, Optional
import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class HybridFCL(nn.Module):
    """
    Classical fully‑connected layer with an evaluation API that supports
    batched parameter sets and optional Gaussian shot noise.

    The forward pass mimics the original FCL example: a linear map followed
    by a tanh non‑linearity and a mean over the batch.  The ``evaluate``
    method is inspired by FastBaseEstimator and FastEstimator, allowing
    a list of observables (functions) to be applied to the output and
    optionally adding noise to emulate finite‑shot effects.
    """

    def __init__(self, n_features: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=bias)

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        """Return a single‑dimensional expectation value."""
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return torch.tanh(self.linear(values)).mean(dim=0)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Compute expectations for a list of parameter sets.

        Parameters
        ----------
        observables
            Iterable of functions that map a model output to a scalar.
            If ``None`` a mean over the last dimension is used.
        parameter_sets
            Sequence of parameter vectors to evaluate.
        shots
            If provided, Gaussian noise with variance 1/shots is added.
        seed
            Random seed for noise generation.
        """
        if parameter_sets is None:
            return []

        observables = list(observables) or [lambda x: x.mean(dim=-1)]
        results: List[List[float]] = []

        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                outputs = self.forward(params)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    scalar = float(val.mean().cpu()) if isinstance(val, torch.Tensor) else float(val)
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


__all__ = ["HybridFCL"]
