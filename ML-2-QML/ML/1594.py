"""Enhanced estimator for PyTorch models with batched evaluation, noise injection, and autodifferentiation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], Union[torch.Tensor, float, List[float]]]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence into a batch of shape (1, D)."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastEstimator:
    """
    Evaluate a PyTorch model for a set of parameter vectors and observables.

    Features
    --------
    * Batch‑wise evaluation of multiple parameter sets.
    * Supports custom scalar or vector observables.
    * Optional Gaussian or Poisson shot noise.
    * Automatic differentiation of model outputs w.r.t. parameters.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.  The model should accept a batch of inputs
        and return a tensor of shape (batch, out_dim).
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        noise: str | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Compute the observables for each set of parameters.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output tensor and returns a scalar
            (or a tensor that will be reduced to a scalar).  If empty a default
            mean over the output dimension is used.
        parameter_sets : sequence of sequences
            Each inner sequence is a list of parameter values for one evaluation.
        shots : int, optional
            If provided, Gaussian shot noise with variance 1/shots is added.
        noise : {'gaussian', 'poisson'}, optional
            Type of stochastic noise to apply.  Noise is applied only if *shots*
            is specified.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        List[List[float]]
            A list of rows, one per parameter set, each containing the value of
            every observable.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().item())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)

        if shots is None or noise is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = []
            for mean in row:
                if noise == "gaussian":
                    noisy_val = rng.normal(mean, max(1e-6, 1 / np.sqrt(shots)))
                elif noise == "poisson":
                    noisy_val = rng.poisson(mean * shots) / shots
                else:
                    raise ValueError(f"Unsupported noise type: {noise}")
                noisy_row.append(noisy_val)
            noisy.append(noisy_row)
        return noisy

    def gradient(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        grad_mode: str = "autograd",
    ) -> List[List[torch.Tensor]]:
        """
        Compute the gradient of each observable w.r.t. the model parameters.

        Parameters
        ----------
        observables : iterable of callables
            As in :meth:`evaluate`.
        parameter_sets : sequence of sequences
            As in :meth:`evaluate`.
        grad_mode : {'autograd', 'finite'}, optional
            Method to compute the gradient.  ``'autograd'`` uses PyTorch's autograd
            while ``'finite'`` uses a central finite difference approximation.

        Returns
        -------
        List[List[torch.Tensor]]
            Gradient tensors for each observable and parameter set.  The shape of
            each gradient tensor matches the shape of the model parameters.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        grads: List[List[torch.Tensor]] = []

        for params in parameter_sets:
            inputs = _ensure_batch(params)
            inputs.requires_grad_(True)
            outputs = self.model(inputs)
            row: List[torch.Tensor] = []
            for observable in observables:
                value = observable(outputs)
                if isinstance(value, torch.Tensor):
                    scalar = value.mean()
                else:
                    scalar = torch.tensor(value, dtype=torch.float32)
                if grad_mode == "autograd":
                    scalar.backward(retain_graph=True)
                elif grad_mode == "finite":
                    eps = 1e-6
                    grad = torch.zeros_like(inputs)
                    for i in range(inputs.numel()):
                        perturbed = inputs.clone()
                        perturbed[0, i] += eps
                        out_plus = self.model(perturbed)
                        val_plus = observable(out_plus).mean()
                        perturbed[0, i] -= 2 * eps
                        out_minus = self.model(perturbed)
                        val_minus = observable(out_minus).mean()
                        grad[0, i] = (val_plus - val_minus) / (2 * eps)
                    scalar = None
                else:
                    raise ValueError(f"Unsupported grad_mode: {grad_mode}")
                row.append(inputs.grad.clone())
                inputs.grad.zero_()
            grads.append(row)
        return grads


__all__ = ["FastEstimator"]
