"""FastBaseEstimator with PyTorch backend and optional quantum-inspired shot noise.

The estimator accepts a PyTorch nn.Module that maps a batch of parameters
to a batch of outputs.  Observables are callables that transform the
model outputs into scalars.  The evaluate method supports an optional
shot noise model that emulates the statistical fluctuations of
quantum measurements.

Example
-------
>>> import torch
>>> from torch import nn
>>> model = nn.Sequential(nn.Linear(1, 10), nn.ReLU(), nn.Linear(10, 1))
>>> est = FastBaseEstimator(model)
>>> obs = [lambda out: out.mean().item()]
>>> params = [[0.5], [1.0], [1.5]]
>>> est.evaluate(obs, params)
[[0.5], [1.0], [1.5]]
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Wrap a 1‑D sequence as a batch of shape (1, N)."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for a list of parameter sets and observables.

    Parameters
    ----------
    model : nn.Module
        A PyTorch neural network that maps a (B, *) input to a tensor of
        shape (B, O).  The model is set to ``eval`` mode during evaluation.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Return a table of observable values for each parameter set.

        Parameters
        ----------
        observables
            Iterable of callables that turn the model output into a
            scalar (or a tensor that will be reduced to a scalar).
        parameter_sets
            Sequence of sequences of floats that will be fed to the model.
        shots
            If supplied, Gaussian noise with variance 1/shots is added to
            each observable to emulate quantum sampling.
        seed
            Random seed for the noise generator.

        Returns
        -------
        List[List[float]]
            A 2‑D list where each row corresponds to a parameter set and each
            column to an observable.
        """
        observables = list(observables) or [lambda out: out.mean()]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)

        # Add shot noise if requested
        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [
                    rng.normal(mean, max(1e-6, 1 / shots)) for mean in row
                ]
                noisy.append(noisy_row)
            return noisy

        return results


def FCL(n_features: int = 1) -> nn.Module:
    """Return a one‑layer fully‑connected PyTorch model.

    The model implements `run(thetas)` for compatibility with the
    quantum FCL example – it simply forwards the parameters through a
    linear layer, applies tanh, and returns the mean.
    """
    class _FCL(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().cpu().numpy()

    return _FCL()


__all__ = ["FastBaseEstimator", "FCL"]
