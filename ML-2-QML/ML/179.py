"""Enhanced estimator utilities implemented with PyTorch modules.

This module extends the original lightweight `FastBaseEstimator` by
providing device awareness, optional gradient evaluation with respect
to the input parameters, and a convenient subclass that adds shot
noise.  The API remains backwards compatible for the `evaluate`
method, while the new `evaluate_gradients` method exposes the
gradient of each observable with respect to the supplied parameter
vectors.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List

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
    model
        A PyTorch ``nn.Module`` that maps a batch of input parameters to
        output tensors.  The module is moved to *device* and cast to
        *dtype* during construction.
    device
        Target device (`'cpu'`, `'cuda'`, or a ``torch.device``).
    dtype
        Numerical type used for the input tensors.
    """

    def __init__(self, model: nn.Module, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        self.model = model.to(device).type(dtype)
        self.device = torch.device(device)
        self.dtype = dtype

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute deterministic expectation values for each parameter set.

        The method is fully compatible with the original API and
        accepts an arbitrary number of callable observables.  Each
        observable is applied to the model output and the result is
        reduced to a scalar.

        Parameters
        ----------
        observables
            Iterable of callables that map the model output to a scalar.
        parameter_sets
            Sequence of parameter vectors (each a sequence of floats).

        Returns
        -------
        List[List[float]]
            A 2‑D list where the outer index runs over parameter sets and
            the inner index over observables.
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
        return results

    def evaluate_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[List[float]]]:
        """Return the gradient of each observable w.r.t. the input parameters.

        The gradient is computed with ``torch.autograd``.  For each
        parameter set a list of observables is produced, and for each
        observable a list of gradients (one per input parameter) is
        returned.

        Parameters
        ----------
        observables
            Iterable of callables that map the model output to a scalar.
        parameter_sets
            Sequence of parameter vectors.

        Returns
        -------
        List[List[List[float]]]
            ``gradients[set_idx][obs_idx][param_idx]`` gives the partial
            derivative of ``observables[obs_idx]`` w.r.t. the
            ``param_idx``‑th input parameter for the ``set_idx``‑th
            parameter set.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        gradients: List[List[List[float]]] = []

        self.model.eval()
        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device, dtype=self.dtype)
            inputs.requires_grad_(True)
            outputs = self.model(inputs)

            grad_set: List[List[float]] = []
            for observable in observables:
                value = observable(outputs)
                if isinstance(value, torch.Tensor):
                    # Ensure a scalar for gradient computation
                    scalar = value.mean()
                else:
                    scalar = torch.tensor(value, dtype=self.dtype, device=self.device)
                grads = torch.autograd.grad(scalar, inputs, retain_graph=True, allow_unused=True)[0]
                # grads shape: (batch, param_dim) -> take first batch element
                grad_vec = grads.squeeze(0).cpu().numpy().tolist()
                grad_set.append(grad_vec)
            gradients.append(grad_set)
        return gradients


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator.

    The noise model is applied after the deterministic evaluation and
    mimics finite‑sample shot noise.  The class is fully backwards
    compatible with the original ``FastEstimator``.
    """

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
