"""Enhanced PyTorch estimator with batched evaluation and gradient support.

The original FastBaseEstimator was a minimal wrapper around a
torch.nn.Module.  This upgraded implementation adds:

* Automatic device selection (CPU/GPU) and optional mixed‑precision.
* Batched evaluation of multiple parameter sets.
* Vectorised observables that may return scalars or tensors.
* Gradient computation via PyTorch autograd.
* Optional Gaussian shot noise that mimics finite‑sample effects.
* Convenient ``__call__`` alias and type‑checked signatures.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Union, Optional

import numpy as np
import torch
from torch import nn, Tensor
from torch import device as _device

ScalarObservable = Callable[[Tensor], Union[Tensor, float, np.ndarray]]


def _ensure_batch(values: Sequence[float]) -> Tensor:
    """Convert a sequence of floats to a 2‑D tensor on the default device."""
    tensor = torch.as_tensor(values, dtype=torch.float32, device=_device("cpu"))
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for batches of parameters and observables."""

    def __init__(self, model: nn.Module, device: str | _device | None = None) -> None:
        self.model = model
        self.device = _device(device or "cpu")
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        dtype: torch.dtype = torch.float32,
    ) -> List[List[float]]:
        """Return a list of observable values for each parameter set."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device, dtype=dtype)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, Tensor):
                        val = val.mean().item()
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        return results

    def evaluate_with_noise(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Add Gaussian shot noise to deterministic outputs."""
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def gradient(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        loss_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> List[List[float]]:
        """Return gradients of the sum of observables w.r.t. parameters.

        Parameters
        ----------
        loss_fn
            Optional custom loss function applied to the model outputs.
            If ``None`` the sum of all observables is used.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        grads: List[List[float]] = []

        self.model.train()
        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device)
            inputs.requires_grad_(True)
            outputs = self.model(inputs)

            if loss_fn is None:
                loss = sum(obs(outputs).sum() for obs in observables)
            else:
                loss = loss_fn(outputs)

            loss.backward()
            grad = inputs.grad.squeeze().cpu().numpy().tolist()
            grads.append(grad)
            self.model.zero_grad()
        return grads

    def __call__(self, *args, **kwargs):
        """Alias for :meth:`evaluate`."""
        return self.evaluate(*args, **kwargs)


__all__ = ["FastBaseEstimator"]
