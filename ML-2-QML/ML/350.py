"""Enhanced lightweight estimator utilities with batch processing, GPU support, and noise simulation."""

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
    """Evaluate neural networks for batches of inputs and observables.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to evaluate.
    device : torch.device | str | None, optional
        Device on which to run the model (default: CPU).
    batch_size : int | None, optional
        Maximum batch size for evaluation.  ``None`` evaluates all at once.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device | str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device or "cpu")
        self.batch_size = batch_size
        self.model.to(self.device)

    def _evaluate_batch(
        self,
        params_batch: torch.Tensor,
        observables: List[ScalarObservable],
    ) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(params_batch.to(self.device))
            results = []
            for obs in observables:
                val = obs(outputs)
                if isinstance(val, torch.Tensor):
                    val = val.mean(dim=-1)
                results.append(val.cpu())
            # shape: (batch, num_obs)
            return torch.stack(results, dim=1)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        return_numpy: bool = True,
        progress: bool = False,
    ) -> List[List[float]] | np.ndarray:
        """Return expectation values for each parameter set and observable.

        The method can return a NumPy array when ``return_numpy=True``.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        param_tensor = torch.as_tensor(parameter_sets, dtype=torch.float32, device=self.device)
        results = []

        if self.batch_size is None or self.batch_size >= param_tensor.shape[0]:
            batch_res = self._evaluate_batch(param_tensor, observables)
            results = batch_res.cpu().numpy()
        else:
            for i in range(0, param_tensor.shape[0], self.batch_size):
                batch = param_tensor[i : i + self.batch_size]
                batch_res = self._evaluate_batch(batch, observables)
                results.append(batch_res.cpu().numpy())

        if return_numpy:
            return np.vstack(results)
        else:
            return results.tolist()

    def evaluate_iter(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        progress: bool = False,
    ) -> Iterable[List[float]]:
        """Yield results one by one, useful for streaming large datasets."""
        for row in self.evaluate(
            observables,
            parameter_sets,
            return_numpy=False,
            progress=progress,
        ):
            yield row


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        return_numpy: bool = True,
        progress: bool = False,
    ) -> List[List[float]] | np.ndarray:
        raw = super().evaluate(
            observables,
            parameter_sets,
            return_numpy=return_numpy,
            progress=progress,
        )
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        if isinstance(raw, np.ndarray):
            shape = raw.shape
            std = np.maximum(1e-6, 1 / np.sqrt(shots))
            noisy = raw + rng.normal(0, std, size=shape)
            return noisy
        else:
            noisy = [
                [float(rng.normal(val, max(1e-6, 1 / np.sqrt(shots)))) for val in row]
                for row in raw
            ]
            return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
