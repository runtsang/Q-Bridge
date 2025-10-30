"""Extended lightweight estimator utilities implemented with PyTorch modules.

This module provides a flexible, GPU‑aware estimator that supports batched
evaluation, configurable observables, optional weighting, and automatic
gradient computation.  It preserves the original API while adding
practical extensions for research workflows.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of floats into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables.

    The estimator is intentionally lightweight: it avoids unnecessary
    memory allocations, supports GPU execution, and can process large
    collections of parameter sets in a single forward pass.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        batch_size: Optional[int] = None,
        weights: Optional[Sequence[float]] = None,
        dropout: Optional[float] = None,
        **kwargs,
    ) -> List[List[float]]:
        """
        Compute the observables for each parameter set.

        Parameters
        ----------
        observables:
            Callables that map model outputs to scalar values.
        parameter_sets:
            Iterable of parameter vectors.
        batch_size:
            Number of samples to evaluate per forward pass.  ``None`` evaluates
            all samples at once.
        weights:
            Optional weights applied to each observable.  Must match the number
            of observables.
        dropout:
            If provided, the model is temporarily set to training mode with
            the given dropout probability for the duration of the call.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        if weights is not None:
            if len(weights)!= len(observables):
                raise ValueError("Length of weights must match number of observables.")
            weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        # Temporarily adjust dropout if requested
        original_mode = self.model.training
        if dropout is not None:
            self.model.train()
            for m in self.model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = dropout

        results: List[List[float]] = []
        param_tensors = torch.as_tensor(parameter_sets, dtype=torch.float32, device=self.device)

        # Process in batches to keep memory usage under control
        if batch_size is None or batch_size <= 0:
            batch_size = len(param_tensors)

        with torch.no_grad():
            for start in range(0, len(param_tensors), batch_size):
                batch = param_tensors[start : start + batch_size]
                outputs = self.model(batch)
                for row_idx, params in enumerate(batch):
                    row: List[float] = []
                    for obs_idx, observable in enumerate(observables):
                        val = observable(outputs[row_idx])
                        if isinstance(val, torch.Tensor):
                            scalar = float(val.mean().cpu())
                        else:
                            scalar = float(val)
                        if weights is not None:
                            scalar *= float(weights[obs_idx].cpu())
                        row.append(scalar)
                    results.append(row)

        # Restore original dropout probability and training mode
        if dropout is not None:
            for m in self.model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = 0.0
        if original_mode:
            self.model.train()
        else:
            self.model.eval()

        return results

    def compute_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        batch_size: Optional[int] = None,
    ) -> List[List[torch.Tensor]]:
        """
        Compute gradients of the observables with respect to the model parameters.

        Returns a list of lists of gradient tensors, mirroring the structure of
        ``evaluate``.  Each gradient tensor has the same shape as the flattened
        model parameters.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        param_tensors = torch.as_tensor(parameter_sets, dtype=torch.float32, device=self.device, requires_grad=True)

        if batch_size is None or batch_size <= 0:
            batch_size = len(param_tensors)

        gradients: List[List[torch.Tensor]] = []

        for start in range(0, len(param_tensors), batch_size):
            batch = param_tensors[start : start + batch_size]
            outputs = self.model(batch)
            for row_idx, params in enumerate(batch):
                grads_row: List[torch.Tensor] = []
                for observable in observables:
                    self.model.zero_grad()
                    val = observable(outputs[row_idx])
                    if isinstance(val, torch.Tensor):
                        val = val.mean()
                    val.backward(retain_graph=True)
                    # Flatten all parameter gradients into a single vector
                    grad_vec = torch.cat([p.grad.view(-1) for p in self.model.parameters()]).clone()
                    grads_row.append(grad_vec)
                gradients.append(grads_row)

        return gradients


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets, **kwargs)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
