"""Enhanced lightweight estimator utilities with GPU support, batched inference, and gradient computation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of floats to a 2‑D tensor on the CPU."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for multiple parameter sets and observables.

    The class automatically moves the model to the requested device and supports
    batched inference.  An optional shot‑noise simulation can be requested via
    the ``shots`` argument.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.
    device : str | torch.device, default="cpu"
        Device on which to run the model.
    dtype : torch.dtype, default=torch.float32
        Tensor data type.
    """

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> None:
        self.model = model.to(device).type(dtype)
        self.device = torch.device(device)
        self.dtype = dtype

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
        observables
            Iterable of callables that map the model output to a scalar.
        parameter_sets
            Sequence of parameter vectors to evaluate.
        shots
            If provided, Gaussian noise with variance ``1/shots`` is added to
            each mean value to mimic shot noise.
        seed
            Random seed for reproducible noise.

        Returns
        -------
        List[List[float]]
            Outer list over parameter sets, inner list over observables.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device, dtype=self.dtype)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def compute_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[List[float]]]:
        """Compute analytical gradients of each observable w.r.t. parameters.

        Returns a nested list:
            [parameter_set][observable][parameter].
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        grads: List[List[List[float]]] = []

        self.model.train()
        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device, dtype=self.dtype)
            inputs.requires_grad_(True)
            outputs = self.model(inputs)

            row_grads: List[List[float]] = []
            for observable in observables:
                value = observable(outputs)
                if isinstance(value, torch.Tensor):
                    scalar = value.mean()
                else:
                    scalar = torch.tensor(value, dtype=self.dtype, device=self.device)
                grad = torch.autograd.grad(scalar, inputs, retain_graph=True, create_graph=False)[0]
                row_grads.append(grad.squeeze(0).cpu().numpy().tolist())
            grads.append(row_grads)

        return grads


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator.

    Inherits all functionality from :class:`FastBaseEstimator` and augments the
    ``evaluate`` method with a ``shots`` argument.
    """


__all__ = ["FastBaseEstimator", "FastEstimator"]
