"""Unified estimator that extends FastBaseEstimator with richer functionality.

This class accepts any PyTorch nn.Module and can evaluate a list of scalar
observables.  It adds optional shot‑based Gaussian noise, per‑parameter scaling,
and GPU support while keeping the original simple interface.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, List, Optional, Sequence

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


class UnifiedEstimator:
    """Evaluate a PyTorch model for multiple parameter sets and observables.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.
    device : str | torch.device, optional
        Device on which to run the model (default: "cpu").
    """

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        noise_std: Optional[float] = None,
        scaling_factors: Optional[Sequence[float]] = None,
    ) -> List[List[float]]:
        """Compute scalar observables for each parameter set.

        The default behaviour reproduces the original FastBaseEstimator
        (deterministic evaluation).  Optional arguments enable
        shot‑based Gaussian noise, a fixed noise standard deviation,
        and per‑parameter scaling.

        Parameters
        ----------
        observables : Iterable[Callable[[torch.Tensor], torch.Tensor | float]]
            Functions that map model outputs to scalars.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors.
        shots : int, optional
            If supplied, Gaussian noise with variance 1/shots is added to each
            observable value to mimic finite‑sample shot noise.
        seed : int, optional
            Random seed for reproducible noise.
        noise_std : float, optional
            Explicit Gaussian noise standard deviation that overrides the
            shot‑based variance.
        scaling_factors : Sequence[float], optional
            Multiplicative factor applied to each parameter vector before
            evaluation.

        Returns
        -------
        List[List[float]]
            A matrix where each row corresponds to a parameter set and each
            column to an observable.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        # Prepare the RNG for optional noise
        rng = np.random.default_rng(seed)

        self.model.eval()
        with torch.no_grad():
            for idx, params in enumerate(parameter_sets):
                # Apply per‑parameter scaling if requested
                if scaling_factors is not None:
                    scale = scaling_factors[idx]
                    params = [p * scale for p in params]
                inputs = _ensure_batch(params).to(self.device)
                outputs = self.model(inputs)

                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    # Convert tensor to float
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)

                # Add Gaussian noise if requested
                if shots is not None or noise_std is not None:
                    sigma = noise_std if noise_std is not None else math.sqrt(1.0 / shots)
                    noisy_row = [rng.normal(mean, max(1e-12, sigma)) for mean in row]
                    row = noisy_row

                results.append(row)

        return results


__all__ = ["UnifiedEstimator"]
