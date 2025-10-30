"""FastBaseEstimator: batched, GPU‑aware estimator for PyTorch models.

This module extends the original lightweight estimator with:
* Batch‑wise evaluation of parameter sets, reducing memory overhead.
* Optional device selection (CPU/GPU) and automatic tensor placement.
* Vectorised application of arbitrary scalar observables.
* Shot‑noise simulation via Gaussian perturbation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

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
    """Evaluate a PyTorch model on batches of parameters with arbitrary observables.

    Parameters
    ----------
    model
        A `torch.nn.Module` that maps a batch of parameters to outputs.
    device
        Optional device; defaults to ``'cpu'`` or the first available GPU.
    """

    def __init__(self, model: nn.Module, device: Optional[Union[str, torch.device]] = None) -> None:
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        batch_size: int = 64,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate the model for each set of parameters.

        Parameters
        ----------
        observables
            Iterable of callables that accept a model output tensor and return a
            scalar or a tensor that can be reduced to a scalar.
        parameter_sets
            Sequence of parameter vectors.
        batch_size
            Number of parameter sets to evaluate per forward pass.
        shots
            If provided, Gaussian noise with variance ``1/shots`` is added to each
            observable to emulate shot noise.
        seed
            Random seed for reproducible noise.

        Returns
        -------
        List[List[float]]
            A 2‑D list where each row corresponds to a parameter set and each
            column corresponds to an observable.
        """
        if not parameter_sets:
            return []

        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]

        # Convert parameter sets to tensor
        params_tensor = torch.as_tensor(parameter_sets, dtype=torch.float32, device=self.device)

        rng = np.random.default_rng(seed)
        results: List[List[float]] = []

        with torch.no_grad():
            for batch_idx in range(0, params_tensor.shape[0], batch_size):
                batch_params = params_tensor[batch_idx : batch_idx + batch_size]
                outputs = self.model(batch_params)

                for i, param in enumerate(batch_params):
                    row: List[float] = []
                    for obs in observables:
                        val = obs(outputs[i])
                        if isinstance(val, torch.Tensor):
                            scalar = float(val.mean().cpu())
                        else:
                            scalar = float(val)
                        if shots is not None:
                            noise = rng.normal(0, np.sqrt(1 / shots))
                            scalar += noise
                        row.append(scalar)
                    results.append(row)

        return results


class FastEstimator(FastBaseEstimator):
    """Add explicit shot‑noise simulation to the base estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Same as :class:`FastBaseEstimator` but with Gaussian shot noise."""
        return super().evaluate(
            observables,
            parameter_sets,
            batch_size=64,
            shots=shots,
            seed=seed,
        )


__all__ = ["FastBaseEstimator", "FastEstimator"]
