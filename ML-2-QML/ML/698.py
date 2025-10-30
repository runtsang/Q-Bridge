"""FastBaseEstimator with extended batch, GPU, and noise features."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Union, Optional

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
ScalarObservable = Callable[[torch.Tensor], Union[torch.Tensor, float]]


def _ensure_batch(values: Union[Sequence[float], torch.Tensor]) -> torch.Tensor:
    """Convert a list of float sequences or a torch tensor into a 2‑D batch tensor."""
    if isinstance(values, torch.Tensor):
        tensor = values
    else:
        tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


# --------------------------------------------------------------------------- #
# Estimator classes
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """
    Evaluate a PyTorch model for a batch of parameter sets and a list of
    callable observables.  Supports GPU execution, complex outputs, and
    optional vectorisation of observables.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None) -> None:
        """
        Parameters
        ----------
        model : nn.Module
            The neural network to evaluate.
        device : torch.device, optional
            Compute device; defaults to CUDA if available.
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self._last_inputs: torch.Tensor | None = None
        self._last_outputs: torch.Tensor | None = None

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Union[Sequence[Sequence[float]], torch.Tensor],
        *,
        batch_size: int | None = None,
        return_numpy: bool = True,
    ) -> List[List[float]] | np.ndarray:
        """
        Compute observable values for each parameter set.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Callables that map a model output tensor to a scalar.
        parameter_sets : Sequence[Sequence[float]] or torch.Tensor
            Batch of parameters to feed to the model.
        batch_size : int, optional
            Process parameters in chunks; useful for large batches.
        return_numpy : bool, default True
            Return a numpy array instead of a nested list.

        Returns
        -------
        List[List[float]] or np.ndarray
            Observable values for each parameter set.
        """
        if not observables:
            observables = [lambda out: out.mean(dim=-1)]

        observables = list(observables)

        param_tensor = _ensure_batch(parameter_sets).to(self.device)

        if batch_size is None or batch_size >= param_tensor.shape[0]:
            batch_tensor = param_tensor
            batches = [batch_tensor]
        else:
            batches = torch.split(param_tensor, batch_size, dim=0)

        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for batch in batches:
                outputs = self.model(batch)
                self._last_inputs = batch
                self._last_outputs = outputs

                for row_idx in range(batch.shape[0]):
                    row: List[float] = []
                    for obs in observables:
                        value = obs(outputs[row_idx])
                        if isinstance(value, torch.Tensor):
                            scalar = float(value.mean().cpu())
                        else:
                            scalar = float(value)
                        row.append(scalar)
                    results.append(row)

        if return_numpy:
            return np.asarray(results, dtype=np.float64)
        return results


class FastEstimator(FastBaseEstimator):
    """
    Adds optional Gaussian shot noise to the deterministic estimator.
    """

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Union[Sequence[Sequence[float]], torch.Tensor],
        *,
        shots: int | None = None,
        seed: int | None = None,
        batch_size: int | None = None,
        return_numpy: bool = True,
    ) -> List[List[float]] | np.ndarray:
        """
        Evaluate with optional shot noise.

        Parameters
        ----------
        shots : int, optional
            Number of shots to simulate; if None, returns raw values.
        seed : int, optional
            Random seed for reproducibility.
        """
        raw = super().evaluate(
            observables,
            parameter_sets,
            batch_size=batch_size,
            return_numpy=False,
        )
        if shots is None:
            if return_numpy:
                return np.asarray(raw, dtype=np.float64)
            return raw

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)

        if return_numpy:
            return np.asarray(noisy, dtype=np.float64)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
