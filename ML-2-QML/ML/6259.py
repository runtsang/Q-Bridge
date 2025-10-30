"""FastBaseEstimator for classical neural‑net models with optional shot noise."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """
    Evaluate a PyTorch `nn.Module` on batches of parameters.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.  The model is expected to accept a
        2‑D tensor of shape `(batch, features)` and return a 2‑D tensor
        `(batch, outputs)`.

    Notes
    -----
    * The estimator is stateless and can be reused across multiple calls.
    * If `shots` is provided in `evaluate`, Gaussian noise with variance
      `1/shots` is added to each mean prediction to emulate measurement
      noise.
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Compute scalar outputs for each parameter set.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Callable(s) that map model output tensors to scalars.
        parameter_sets : Sequence[Sequence[float]]
            List of parameter vectors to evaluate.
        shots : int, optional
            If provided, Gaussian noise with variance `1/shots` is added.
        seed : int, optional
            Seed for the random number generator used in noise injection.

        Returns
        -------
        List[List[float]]
            Nested list of results: outer list over parameter sets,
            inner list over observables.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        # Enable GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        rng = np.random.default_rng(seed) if shots is not None else None

        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(device)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)

                if shots is not None:
                    # Add shot‑noise to each mean prediction
                    noisy_row = [
                        float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
                    ]
                    row = noisy_row

                results.append(row)

        return results


__all__ = ["FastBaseEstimator"]
