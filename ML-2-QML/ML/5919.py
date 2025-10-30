from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Callable, Iterable, List, Sequence, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D input sequence into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class EstimatorQNN(nn.Module):
    """Feed‑forward regression network with batched evaluation and optional shot noise.

    The architecture mirrors the original example but accepts a list of hidden
    layer sizes so that the user can experiment with depth and width.  The
    ``evaluate`` method is inspired by the FastBaseEstimator utilities
    from FastBaseEstimator.py and can return deterministic predictions or
    noisy samples that emulate quantum shot noise.

    Parameters
    ----------
    input_dim : int
        Size of the input vector.
    hidden_dims : Sequence[int]
        Sizes of hidden layers.  If empty a linear layer is used.
    """

    def __init__(self, input_dim: int = 2, hidden_dims: Sequence[int] = (8, 4)) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.net(inputs)

    def evaluate(
        self,
        inputs: Sequence[Sequence[float]],
        observables: Iterable[ScalarObservable] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Return predictions (or noisy predictions) for a batch of inputs.

        Parameters
        ----------
        inputs
            Iterable of input vectors.
        observables
            Optional list of scalar observables that are applied to the
            network output.  When ``None`` the mean over the last dimension
            is used, matching the behaviour of the original EstimatorQNN.
        shots
            If provided, Gaussian noise with standard deviation ``1/√shots``
            is added to each prediction to emulate shot noise.
        seed
            Random seed for reproducible noise.

        Returns
        -------
        List[List[float]]
            Nested list where each inner list contains the evaluation of the
            observables for a single input vector.
        """
        if observables is None:
            observables = [lambda out: out.mean(dim=-1)]

        self.eval()
        with torch.no_grad():
            batch = torch.as_tensor(inputs, dtype=torch.float32)
            if batch.ndim == 1:
                batch = batch.unsqueeze(0)
            outputs = self(batch)
            results: List[List[float]] = []
            for out in outputs:
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    scalar = float(val.mean().cpu()) if isinstance(val, torch.Tensor) else float(val)
                    row.append(scalar)
                results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy.append([float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row])
            return noisy

        return results


__all__ = ["EstimatorQNN"]
