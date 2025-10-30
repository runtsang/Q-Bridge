"""Enhanced FastBaseEstimator for PyTorch models with batched inference and optional noise.

Features:
* GPU/CPU device selection.
* Batched parameter evaluation.
* Optional Gaussian shot noise.
* Gradient computation via torch.autograd.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, Iterable, List, Union

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
    """Evaluate a PyTorch neural network for batched parameters and observables."""

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)

    def _forward(self, params: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(params.to(self.device))

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables:
            Callables that map a model output tensor to a scalar or tensor.
        parameter_sets:
            Iterable of parameter sequences.
        shots:
            If provided, Gaussian noise with std = 1/√shots is added to each mean.
        seed:
            Random seed for reproducibility of shot noise.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        rng = np.random.default_rng(seed)
        results: List[List[float]] = []

        batch = torch.cat([_ensure_batch(p) for p in parameter_sets], dim=0)
        outputs = self._forward(batch)

        for row_idx, params in enumerate(parameter_sets):
            out = outputs[row_idx]
            row: List[float] = []
            for obs in observables:
                val = obs(out)
                if isinstance(val, torch.Tensor):
                    scalar = float(val.mean().cpu())
                else:
                    scalar = float(val)
                row.append(scalar)
            if shots is not None:
                std = max(1e-6, 1 / np.sqrt(shots))
                row = [float(rng.normal(s, std)) for s in row]
            results.append(row)
        return results

    def grad(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[np.ndarray]]:
        """Return gradients of observables w.r.t. parameters for each set."""
        self.model.train()
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        grads: List[List[np.ndarray]] = []

        for params in parameter_sets:
            params_tensor = torch.tensor(
                params, dtype=torch.float32, requires_grad=True, device=self.device
            )
            out = self.model(params_tensor.unsqueeze(0))
            for obs in observables:
                val = obs(out)
                if isinstance(val, torch.Tensor):
                    scalar = val.mean()
                else:
                    scalar = torch.tensor(val, dtype=torch.float32, device=self.device)
                scalar.backward()
                grads.append(params_tensor.grad.cpu().numpy().copy())
                params_tensor.grad.zero_()
        return grads


__all__ = ["FastBaseEstimator"]
