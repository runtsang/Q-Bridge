"""Hybrid classical estimator with noise support and preprocessing.

This module defines a lightweight, batched evaluator that works with any
PyTorch ``nn.Module``.  It supports:

* Automatic batch handling of scalar or vector parameters.
* Optional preprocessing of input tensors before they reach the model.
* Shot‑noise simulation by adding Gaussian noise with a user‑supplied
  ``shots`` count.
* A small, drop‑in subclass that exposes the same API but always returns
  noisy measurements.

The implementation is intentionally minimal to keep the estimator
fast and easy to embed in larger pipelines.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[Union[float, Sequence[float]]]) -> torch.Tensor:
    """Convert raw parameter lists to a 2‑D ``torch.Tensor``."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for a batch of parameter sets.

    Parameters
    ----------
    model
        Any ``torch.nn.Module`` that accepts a ``torch.Tensor`` of
        shape ``(batch, features)`` and returns a tensor of shape
        ``(batch, outputs)``.
    device
        Target device for the model and tensors.
    dtype
        Data type for the input tensors.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.dtype = dtype

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | torch.Tensor | None = None,
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> List[List[float]]:
        """Compute observables for each parameter set.

        Parameters
        ----------
        observables
            Iterable of callables that map the model output to a scalar.
            If ``None`` a mean over the last dimension is used.
        parameter_sets
            Iterable of parameter sequences or a tensor of shape
            ``(batch, features)``.
        shots
            If provided, the deterministic predictions are perturbed
            with Gaussian noise of variance ``1/shots``.
        seed
            Random seed for the noise generator.
        preprocess
            Optional transformation applied to the batch before the model
            forward pass.
        """
        if observables is None:
            observables = [lambda outputs: outputs.mean(dim=-1)]
        observables = list(observables)

        if parameter_sets is None:
            return []

        params = torch.as_tensor(parameter_sets, dtype=self.dtype, device=self.device)
        if params.ndim == 1:
            params = params.unsqueeze(0)

        if preprocess is not None:
            params = preprocess(params)

        with torch.no_grad():
            outputs = self.model(params)

        results: List[List[float]] = []
        for row in outputs:
            row_vals: List[float] = []
            for obs in observables:
                val = obs(row)
                if isinstance(val, torch.Tensor):
                    val = val.mean().item()
                row_vals.append(float(val))
            results.append(row_vals)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [rng.normal(mean, max(1e-6, 1 / shots)) for mean in row]
            noisy.append(noisy_row)
        return noisy


class FastEstimator(FastBaseEstimator):
    """Subclass that always returns noisy measurements.

    The constructor is identical to :class:`FastBaseEstimator`.  The
    ``evaluate`` method simply forwards all arguments and forces a
    ``shots`` value if none was supplied.
    """

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | torch.Tensor | None = None,
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> List[List[float]]:
        if shots is None:
            shots = 1_000  # default to a reasonably high precision
        return super().evaluate(
            observables=observables,
            parameter_sets=parameter_sets,
            shots=shots,
            seed=seed,
            preprocess=preprocess,
        )


__all__ = ["FastBaseEstimator", "FastEstimator"]
