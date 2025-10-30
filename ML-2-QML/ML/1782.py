"""Enhanced lightweight estimator utilities built on PyTorch.

Features
--------
* ``evaluate`` now accepts *broadcastable* parameter arrays and returns a NumPy array for easier downstream use.
* ``evaluate_gradients`` returns the Jacobian of each observable with respect to the input parameters.
* Optional Gaussian shot‑noise can be applied during evaluation via ``shots`` keyword.
* The class implements ``__call__`` as a shorthand for ``evaluate``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.autograd import grad

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a list of parameter lists into a 2‑D tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        device: str | torch.device | None = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        observables : iterable of callables
            Each callable receives a ``torch.Tensor`` of model outputs and
            must return either a scalar ``torch.Tensor`` or a Python float.
        parameter_sets : sequence of sequences of float
            Batch of parameter vectors to feed into the model.
        device : optional
            Torch device to run inference on.  If ``None`` the model's
            current device is used.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, n_observables)``.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        self.model.eval()
        batch = _ensure_batch(parameter_sets).to(device or self.model.parameters().__next__().device)
        with torch.no_grad():
            outputs = self.model(batch)
        results = []
        for obs in observables:
            value = obs(outputs)
            if isinstance(value, torch.Tensor):
                scalar = value.squeeze().detach().cpu().numpy()
            else:
                scalar = np.array(value)
            results.append(scalar)
        return np.stack(results, axis=-1)

    def evaluate_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        device: str | torch.device | None = None,
    ) -> np.ndarray:
        """
        Returns the Jacobian ``∂O/∂θ`` for every observable and parameter set.
        Uses autograd to compute gradients.

        The output shape is ``(n_samples, n_observables, n_params)``.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        batch = _ensure_batch(parameter_sets).to(device or self.model.parameters().__next__().device)
        batch.requires_grad_(True)
        self.model.eval()
        outputs = self.model(batch)

        jacobians = []
        for obs in observables:
            value = obs(outputs).sum()
            grads = grad(value, batch, retain_graph=True, create_graph=False)[0]
            jacobians.append(grads.detach().cpu().numpy())
        return np.stack(jacobians, axis=1)

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def _with_shot_noise(
        self,
        raw: np.ndarray,
        shots: int | None,
        seed: int | None,
    ) -> np.ndarray:
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0, scale=1.0 / np.sqrt(shots), size=raw.shape)
        return raw + noise


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        device: str | torch.device | None = None,
    ) -> np.ndarray:
        raw = super().evaluate(observables, parameter_sets, device=device)
        return self._with_shot_noise(raw, shots, seed)

    def evaluate_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        device: str | torch.device | None = None,
    ) -> np.ndarray:
        raw = super().evaluate_gradients(observables, parameter_sets, device=device)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noise = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(shots),
            size=raw.shape,
        )
        return raw + noise


__all__ = ["FastBaseEstimator", "FastEstimator"]
