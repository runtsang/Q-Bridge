"""UniversalEstimator: a hybrid classical estimator with batched evaluation, shot‑noise simulation, and gradient support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn
from torch.autograd import grad

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class UniversalEstimator:
    """Evaluate neural networks for batches of inputs and observables with optional shot noise and gradient support."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Compute observables for each parameter set. Returns a NumPy array of shape (n_sets, n_observables)."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        self.model.eval()
        with torch.no_grad():
            params_tensor = torch.as_tensor(parameter_sets, dtype=torch.float32)
            if params_tensor.ndim == 1:
                params_tensor = params_tensor.unsqueeze(0)
            outputs = self.model(params_tensor)
        results: List[List[float]] = []
        for i in range(outputs.shape[0]):
            row: List[float] = []
            for obs in observables:
                val = obs(outputs[i:i+1])
                if isinstance(val, torch.Tensor):
                    scalar = float(val.mean().item())
                else:
                    scalar = float(val)
                row.append(scalar)
            results.append(row)
        arr = np.array(results, dtype=float)
        if shots is not None:
            rng = np.random.default_rng(seed)
            noise = rng.normal(0, 1 / np.sqrt(shots), size=arr.shape)
            arr += noise
        return arr

    def gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Compute gradients of observables w.r.t. parameters for each parameter set."""
        observables = list(observables)
        grads_list: List[List[float]] = []
        self.model.train()
        for params in parameter_sets:
            params_tensor = torch.as_tensor(params, dtype=torch.float32, requires_grad=True).unsqueeze(0)
            outputs = self.model(params_tensor)
            for obs in observables:
                val = obs(outputs)
                if isinstance(val, torch.Tensor):
                    val = val.mean()
                grad_val = grad(val, params_tensor, retain_graph=True)[0]
                grads_list.append(grad_val.squeeze().cpu().numpy())
        return np.array(grads_list)

__all__ = ["UniversalEstimator"]
