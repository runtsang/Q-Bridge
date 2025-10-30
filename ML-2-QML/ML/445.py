"""Enhanced lightweight estimator utilities with batched evaluation, GPU support, and gradient computation."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Dict, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch neural network for a set of parameter vectors.

    Parameters
    ----------
    model: nn.Module
        The neural network to evaluate.  The model must accept a batch tensor of shape
        ``(batch, input_dim)`` and return a tensor of arbitrary shape.  The model is
        evaluated in ``eval`` mode and gradients are disabled.
    device: str | torch.device, optional
        Device on which the model and data are placed.  ``'cpu'`` is the default;
        ``'cuda'`` will be used automatically if available.
    cache: bool, optional
        If ``True`` (default) the outputs for each unique parameter vector are cached
        to avoid recomputation when the same set of parameters is queried repeatedly.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
        *,
        cache: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.model.eval()
        self._cache: Dict[Tuple[float,...], torch.Tensor] = {}
        self._cache_enabled = cache

    def _forward(self, params: torch.Tensor) -> torch.Tensor:
        """Run the model once, using the cache if enabled."""
        key = tuple(params.tolist())
        if self._cache_enabled and key in self._cache:
            return self._cache[key]
        with torch.no_grad():
            outputs = self.model(params.to(self.device))
        if self._cache_enabled:
            self._cache[key] = outputs.cpu()
        return outputs.cpu()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float] | torch.Tensor],
    ) -> List[List[float]]:
        """Compute the scalar expectation values for each observable and parameter set.

        Parameters
        ----------
        observables : Iterable[Callable[[Tensor], Tensor | float]]
            Functions that map the model output to a scalar.  They are applied to the
            model output for each batch element.  If no observables are supplied a
            default ``mean`` over the last dimension is used.
        parameter_sets : Sequence[Sequence[float] | Tensor]
            Each element is a list of parameter values or a 1‑D tensor.  All
            parameter vectors must have the same dimensionality.
        Returns
        -------
        List[List[float]]
            A list of rows, one per parameter set, where each row contains the
            scalar value for each observable.
        """
        if not observables:
            observables = [lambda out: out.mean(dim=-1)]
        observables = list(observables)

        results: List[List[float]] = []
        for params in parameter_sets:
            if isinstance(params, torch.Tensor):
                batch = params
            else:
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
            results.append(row)
        return results

    def gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[torch.Tensor]]:
        """Return gradients of each observable w.r.t. the model parameters.

        The returned gradients are given in the same order as the input
        ``parameter_sets`` and ``observables``.  Each gradient is a tensor of shape
        ``(num_params,)``.
        """
        if not observables:
            observables = [lambda out: out.mean(dim=-1)]
        observables = list(observables)

        grads: List[List[torch.Tensor]] = []
        for params in parameter_sets:
            batch = _ensure_batch(params)
            batch = batch.to(self.device).requires_grad_(True)
            outputs = self.model(batch)
            for obs in observables:
                val = obs(outputs)
                if isinstance(val, torch.Tensor):
                    val = val.mean()
                grad = torch.autograd.grad(val, self.model.parameters(), retain_graph=True)
                # Concatenate all parameter gradients into a single vector
                grad_vec = torch.cat([g.reshape(-1) for g in grad if g is not None])
                grads.append([grad_vec.cpu()])
        return grads


class FastEstimator(FastBaseEstimator):
    """Same as FastBaseEstimator but adds optional shot‑noise simulation.

    Parameters
    ----------
    model: nn.Module
        The neural network to evaluate.
    device: str | torch.device, optional
        Device for computation.
    cache: bool, optional
        Enable caching of intermediate outputs.
    noise_dist: str, optional
        Distribution used for shot noise.  ``'gaussian'`` (default) or ``'poisson'``.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
        *,
        cache: bool = True,
        noise_dist: str = "gaussian",
    ) -> None:
        super().__init__(model, device, cache=cache)
        if noise_dist not in {"gaussian", "poisson"}:
            raise ValueError("noise_dist must be 'gaussian' or 'poisson'")
        self.noise_dist = noise_dist

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate with optional shot‑noise simulation.

        Parameters
        ----------
        observables, parameter_sets : same as in FastBaseEstimator
        shots : int, optional
            Number of shots to use for the noise model.  If ``None`` the deterministic
            expectation values are returned.
        seed : int, optional
            Random seed for reproducibility.
        """
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            if self.noise_dist == "gaussian":
                noisy_row = [
                    float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
                ]
            else:  # poisson
                noisy_row = [
                    float(rng.poisson(mean * shots) / shots) for mean in row
                ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
