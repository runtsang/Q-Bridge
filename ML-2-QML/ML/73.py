"""Advanced neural‑network estimator with gradient support.

Provides batch evaluation of neural networks for arbitrary scalar
observables and analytic gradients with respect to model parameters.
Supports GPU execution and optional dropout/normalization layers.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

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
    """Evaluate a PyTorch model for many parameter sets and observables,
    and compute analytic gradients with respect to the model parameters.
    """

    def __init__(self, model: nn.Module, device: Union[str, torch.device] = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a list of observable values for each parameter set."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    scalar = float(value.mean().cpu()) if isinstance(value, torch.Tensor) else float(value)
                    row.append(scalar)
                results.append(row)
        return results

    def evaluate_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[np.ndarray]]:
        """Return gradients of each observable w.r.t. all model parameters
        for every parameter set.  The gradient vector is flattened into
        a 1‑D NumPy array for each observable.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        grads_list: List[List[np.ndarray]] = []
        # No need to set self.model.train(); autograd.grad works on eval mode.
        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device)
            outputs = self.model(inputs)
            row_grads: List[np.ndarray] = []
            for observable in observables:
                value = observable(outputs)
                scalar = value.mean() if isinstance(value, torch.Tensor) else torch.tensor(value, device=self.device, dtype=torch.float32)
                grads = torch.autograd.grad(scalar, self.model.parameters(), retain_graph=True)
                # Flatten all parameter gradients into a single vector
                grad_vec = torch.cat([g.reshape(-1) for g in grads], dim=0).cpu().numpy()
                row_grads.append(grad_vec)
            grads_list.append(row_grads)
        return grads_list


__all__ = ["FastBaseEstimator"]
