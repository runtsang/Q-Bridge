"""Enhanced lightweight estimator utilities built on PyTorch.

The module defines two classes:

* **FastBaseEstimator** – evaluates a PyTorch model for many
  parameter sets and arbitrary scalar observables.  It supports
  batched evaluation, GPU execution, and is fully backward
  compatible with the original seed.

* **FastEstimator** – extends FastBaseEstimator by adding
  Gaussian shot‑noise simulation and a convenient gradient‑
  computation interface based on PyTorch autograd.

Both classes expose a clean API that can be used in research
pipelines or for rapid prototyping.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn


ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence of parameters to a batch of shape (1, N)."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """
    Evaluate a PyTorch model for a collection of parameter sets and
    scalar observables.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.  The model must accept a
        2‑D tensor of shape ``(batch, features)`` and return a tensor
        of arbitrary shape.  The output is passed unchanged to the
        provided observables.

    device : torch.device | str | None, optional
        The device on which to run the model.  Defaults to CPU.

    Notes
    -----
    * The class is fully device‑agnostic; the model will be moved
      to the specified device.
    * ``evaluate`` accepts a list of callable observables.  Each
      observable receives the model output and must return a scalar
      or a 0‑D tensor.
    """

    def __init__(self, model: nn.Module, device: torch.device | str | None = None) -> None:
        self.model = model
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        batch_size: int | None = None,
    ) -> List[List[float]]:
        """
        Compute the observables for every parameter set.

        Parameters
        ----------
        observables : Iterable[Callable]
            Callables that map the model output to a scalar.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors.
        batch_size : int, optional
            If provided, parameters are evaluated in batches of this
            size to keep memory usage bounded.  The default is to
            evaluate all parameters in a single batch.

        Returns
        -------
        List[List[float]]
            A list of rows, one per parameter set, each containing the
            values of all observables.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        # Move the model into evaluation mode
        self.model.eval()
        with torch.no_grad():
            # Flatten all parameters into a 2‑D tensor
            all_params = torch.as_tensor(parameter_sets, dtype=torch.float32, device=self.device)

            if batch_size is None or batch_size >= all_params.shape[0]:
                batches = [all_params]
            else:
                batches = [
                    all_params[i : i + batch_size] for i in range(0, all_params.shape[0], batch_size)
                ]

            for batch in batches:
                outputs = self.model(batch)  # shape (batch,...)

                for out in outputs:
                    row: List[float] = []
                    for obs in observables:
                        value = obs(out)
                        if isinstance(value, torch.Tensor):
                            scalar = float(value.mean().cpu())
                        else:
                            scalar = float(value)
                        row.append(scalar)
                    results.append(row)

        return results


class FastEstimator(FastBaseEstimator):
    """
    Extends FastBaseEstimator with shot‑noise simulation and gradient
    support.

    Methods
    -------
    evaluate
        Same as in FastBaseEstimator, but can add Gaussian noise to the
        outputs to mimic a finite‑shot measurement.
    compute_gradients
        Returns the gradient of each observable with respect to the
        input parameters for each parameter set.
    """

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        batch_size: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate the model and optionally add Gaussian shot noise.

        Parameters
        ----------
        shots : int, optional
            If specified, Gaussian noise with standard deviation
            ``1/√shots`` is added to each observable value.
        seed : int, optional
            Random seed for reproducibility.
        batch_size : int, optional
            Batching strategy forwarded to the base class.
        """
        raw = super().evaluate(observables, parameter_sets, batch_size=batch_size)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def compute_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        batch_size: int | None = None,
    ) -> List[List[np.ndarray]]:
        """
        Compute the gradient of each observable with respect to the input
        parameters for every parameter set.

        Returns
        -------
        List[List[np.ndarray]]
            Outer list indexed by parameter set, inner list indexed by
            observable.  Each element is a NumPy array of shape
            ``(N,)`` where ``N`` is the number of input parameters.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]

        grads: List[List[np.ndarray]] = []

        # Put model in evaluation mode; gradients are computed w.r.t. inputs
        self.model.eval()
        for values in parameter_sets:
            inputs = torch.as_tensor(values, dtype=torch.float32, device=self.device, requires_grad=True)

            outputs = self.model(inputs)
            row_grads: List[np.ndarray] = []
            for obs in observables:
                value = obs(outputs)
                if isinstance(value, torch.Tensor):
                    scalar = value.mean()
                else:
                    scalar = torch.tensor(value, dtype=torch.float32, device=self.device)
                grad = torch.autograd.grad(scalar, inputs, retain_graph=True, allow_unused=True)[0]
                row_grads.append(grad.detach().cpu().numpy())
            grads.append(row_grads)

        return grads


__all__ = ["FastBaseEstimator", "FastEstimator"]
