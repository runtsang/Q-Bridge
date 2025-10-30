"""Hybrid estimator for classical neural network models with optional shot noise and caching."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Dict, Tuple, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridEstimator:
    """Evaluate a PyTorch `nn.Module` for many parameter sets and observables.

    The estimator supports:
    * Vectorised evaluation of a batch of parameters.
    * Optional shot‑noise emulation via Gaussian fluctuations.
    * Caching of identical parameter vectors to avoid redundant forward passes.
    * Custom observables defined as callables that operate on the model output.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._cache: Dict[Tuple[float,...], torch.Tensor] = {}

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        # Cache only when evaluating a single sample
        if batch.shape[0] == 1:
            key = tuple(batch.squeeze(0).tolist())
            if key in self._cache:
                return self._cache[key]
        self.model.eval()
        with torch.no_grad():
            out = self.model(batch)
        if batch.shape[0] == 1:
            self._cache[key] = out
        return out

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Return a list of observable values for each parameter set.

        Parameters
        ----------
        observables:
            Iterable of callables that map a model output tensor to a scalar.
        parameter_sets:
            Iterable of parameter vectors to evaluate.
        shots:
            If provided, add Gaussian noise with std = 1/√shots to each
            observable value to emulate shot noise.
        seed:
            Random seed for noise generation.

        Returns
        -------
        List[List[float]]
            Outer list over parameter sets, inner list over observables.
        """
        if not observables:
            observables = [lambda outputs: float(outputs.mean().cpu())]
        observables = list(observables)

        rng = np.random.default_rng(seed)
        results: List[List[float]] = []

        for params in parameter_sets:
            batch = _ensure_batch(params)
            outputs = self._forward(batch)
            row: List[float] = []
            for obs in observables:
                val = obs(outputs)
                if isinstance(val, torch.Tensor):
                    scalar = float(val.mean().cpu())
                else:
                    scalar = float(val)
                row.append(scalar)

            if shots is not None:
                std = max(1e-6, 1 / np.sqrt(shots))
                row = [float(rng.normal(v, std)) for v in row]

            results.append(row)

        return results

    def evaluate_batch(
        self,
        observables: Iterable[ScalarObservable],
        parameter_tensor: torch.Tensor,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Vectorised evaluation returning a torch.Tensor of shape
        (n_params, n_observables).

        Parameters
        ----------
        parameter_tensor:
            Tensor of shape (n_params, param_dim).
        """
        if not observables:
            observables = [lambda outputs: float(outputs.mean().cpu())]
        observables = list(observables)

        rng = np.random.default_rng(seed) if shots is not None else None
        std = max(1e-6, 1 / np.sqrt(shots)) if shots is not None else None

        outputs = self._forward(parameter_tensor)
        results = []
        for obs in observables:
            val = obs(outputs)
            if isinstance(val, torch.Tensor):
                scalar = val.mean(dim=-1)
            else:
                scalar = torch.full((parameter_tensor.shape[0],), float(val))
            results.append(scalar)

        out_tensor = torch.stack(results, dim=1)  # (n_params, n_observables)

        if shots is not None:
            noise = torch.from_numpy(rng.normal(0, std, size=out_tensor.shape)).float()
            out_tensor = out_tensor + noise

        return out_tensor


__all__ = ["HybridEstimator"]
