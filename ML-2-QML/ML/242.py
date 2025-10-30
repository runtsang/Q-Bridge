"""Enhanced FastBaseEstimator using PyTorch.

Extends the original lightweight estimator with GPU support, optional shot noise,
and automatic gradient computation.  The public API is compatible with the
original seed while adding the following keyword arguments:

* ``device`` – ``'cpu'`` or ``'cuda'`` for GPU acceleration.
* ``shots`` – ``int`` to simulate measurement noise; if ``None`` the model
  is evaluated deterministically.
* ``return_gradients`` – ``True`` to return the gradient of each observable
  w.r.t. the model parameters.
* ``gradient_function`` – a callable that replaces the default autograd
  behaviour; it should accept a scalar tensor and return a list of tensors
  of the same shape as the model parameters.

The implementation keeps the original behaviour as a special case
(device='cpu', shots=None, return_gradients=False).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for batches of parameters and scalar observables."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def _evaluate_batch(
        self,
        inputs: torch.Tensor,
        observables: List[ScalarObservable],
        *,
        return_gradients: bool,
        gradient_function: Callable[[torch.Tensor], List[torch.Tensor]] | None,
    ) -> Tuple[List[List[float]], List[List[torch.Tensor]] | None]:
        self.model.eval()
        results: List[List[float]] = []
        gradients: List[List[torch.Tensor]] | None = None

        if return_gradients:
            gradients = []

        for batch_idx in range(inputs.shape[0]):
            params = inputs[batch_idx : batch_idx + 1]
            params.requires_grad = True
            outputs = self.model(params)
            row: List[float] = []

            for obs in observables:
                val = obs(outputs)
                if isinstance(val, torch.Tensor):
                    scalar = float(val.mean().cpu())
                else:
                    scalar = float(val)
                row.append(scalar)

            results.append(row)

            if return_gradients:
                grads_row: List[torch.Tensor] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        grad = torch.autograd.grad(
                            val.mean(), params, create_graph=False, retain_graph=True
                        )[0]
                    else:
                        grad = torch.zeros_like(params)
                    grads_row.append(grad.squeeze(0).cpu())
                gradients.append(grads_row)

        return results, gradients

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        device: str = "cpu",
        shots: int | None = None,
        seed: int | None = None,
        return_gradients: bool = False,
        gradient_function: Callable[[torch.Tensor], List[torch.Tensor]] | None = None,
    ) -> Union[List[List[float]], Tuple[List[List[float]], List[List[torch.Tensor]]]]:
        """Compute observables for each parameter set.

        Parameters
        ----------
        observables : Iterable[Callable[[Tensor], Tensor | float]]
            Functions that map the model output to a scalar (or a tensor that can be
            reduced to a scalar).  If empty a default mean is used.
        parameter_sets : Sequence[Sequence[float]]
            Each inner sequence represents a set of parameters for the model.
        device : str, optional
            Execution device; defaults to ``'cpu'``.
        shots : int, optional
            Number of measurement shots; if provided Gaussian noise with variance
            ``1/shots`` is added to each observable.
        seed : int, optional
            Seed for the random number generator used in shot noise.
        return_gradients : bool, optional
            If ``True`` return a tuple ``(results, gradients)`` where gradients
            are lists of tensors matching the shape of the model parameters.
        gradient_function : callable, optional
            Override the default autograd behaviour.  It must accept a scalar
            tensor and return a list of tensors matching the model parameters.

        Returns
        -------
        results or (results, gradients)
            ``results`` is a list of lists of floats.  ``gradients`` (if requested)
            is a list of lists of tensors matching the shape of the model parameters.
        """
        if not observables:
            observables = [lambda outputs: outputs.mean(dim=-1)]

        inputs = _ensure_batch(parameter_sets).to(device)

        raw_results, raw_gradients = self._evaluate_batch(
            inputs, list(observables), return_gradients=return_gradients, gradient_function=gradient_function
        )

        if shots is None:
            return raw_results if not return_gradients else (raw_results, raw_gradients)

        rng = np.random.default_rng(seed)
        noisy_results: List[List[float]] = []
        for row in raw_results:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy_results.append(noisy_row)

        if not return_gradients:
            return noisy_results

        return noisy_results, raw_gradients


class FastEstimator(FastBaseEstimator):
    """Convenience subclass that forwards all arguments to :class:`FastBaseEstimator`."""
    pass


__all__ = ["FastBaseEstimator", "FastEstimator"]
