"""Enhanced estimator utilities built on PyTorch.

This module extends the original lightweight estimator by adding:
* batched evaluation and GPU support
* optional shot‑noise simulation
* support for arbitrary callable observables
* simple gradient extraction via autograd
* caching of the last evaluation to avoid redundant forward passes.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Callable, Iterable, List, Sequence, Tuple, Dict

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float], device: torch.device | None = None) -> torch.Tensor:
    """Convert a sequence of scalars into a 2‑D float tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """
    Evaluate a PyTorch model for a collection of parameter sets and observables.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.  It is moved to ``device`` if provided.
    device : str | torch.device | None, optional
        Device on which the model and tensors are placed.  ``None`` uses the default
        device of the model.
    cache : bool, default=True
        When enabled the last forward pass is cached and reused if the same
        parameters are queried again.
    """

    def __init__(self, model: nn.Module, device: torch.device | None = None, cache: bool = True) -> None:
        self.model = model
        if device is not None:
            self.model.to(device)
        self.device = device or next(model.parameters()).device
        self._cache: Dict[Tuple[float,...], torch.Tensor] = {}
        self._cache_enabled = cache

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass, using the cache if possible."""
        key = tuple(inputs.tolist()[0])  # only works for 1‑D batch
        if self._cache_enabled and key in self._cache:
            return self._cache[key]
        self.model.eval()
        with torch.no_grad():
            out = self.model(inputs.to(self.device))
        if self._cache_enabled:
            self._cache[key] = out.cpu()
        return out.cpu()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Compute scalar observables for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output tensor and must return a
            scalar or a tensor that can be reduced to a scalar.
        parameter_sets : sequence of parameter sequences
            Each inner sequence represents a single batch of input values.

        Returns
        -------
        results : list of list of floats
            Outer dimension corresponds to parameter sets, inner dimension to
            observables.
        """
        obs = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        for params in parameter_sets:
            inputs = _ensure_batch(params, device=self.device)
            outputs = self._forward(inputs)
            row: List[float] = []
            for obs_fn in obs:
                val = obs_fn(outputs)
                if isinstance(val, torch.Tensor):
                    scalar = float(val.mean().cpu())
                else:
                    scalar = float(val)
                row.append(scalar)
            results.append(row)

        return results

    # ------------------------------------------------------------------
    # Additional utilities
    # ------------------------------------------------------------------
    def evaluate_with_shots(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Wrap ``evaluate`` and optionally add Gaussian shot‑noise.

        Parameters
        ----------
        shots : int or None
            If provided, each value is perturbed by N(0, 1/√shots).
        seed : int or None
            Random seed for reproducible noise.
        """
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[List[float]]]:
        """
        Compute gradients of each observable w.r.t. all model parameters.

        Returns a 3‑D list: [param_set][observable][parameter].
        """
        obs = list(observables) or [lambda out: out.mean(dim=-1)]
        grads_output: List[List[List[float]]] = []

        for params in parameter_sets:
            inputs = _ensure_batch(params, device=self.device)
            inputs.requires_grad_(True)
            outputs = self.model(inputs)
            param_grads: List[List[float]] = []
            for obs_fn in obs:
                val = obs_fn(outputs)
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val, device=outputs.device)
                else:
                    val = val.mean()
                grads = torch.autograd.grad(val, self.model.parameters(), retain_graph=True)
                grad_flat = [g.detach().cpu().numpy().flatten().tolist() for g in grads]
                param_grads.append(grad_flat)
            grads_output.append(param_grads)

        return grads_output

__all__ = ["FastBaseEstimator"]
