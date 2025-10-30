"""Enhanced estimator utilities for PyTorch models."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Dict, Any

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
ReductionFunction = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float], device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate neural networks for batched inputs and observables with optional device selection."""

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None) -> None:
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        batch_size: int = 64,
        reductions: Optional[Dict[ScalarObservable, ReductionFunction]] = None,
        grad: bool = False,
    ) -> List[List[float]]:
        """
        Compute observables for each parameter set.
        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Callables taking model output and returning a scalar or tensor.
        parameter_sets : Sequence[Sequence[float]]
            Parameter vectors to evaluate.
        batch_size : int, optional
            Chunk size for batched forward passes.
        reductions : dict, optional
            Mapping of observable to a custom reduction function.
        grad : bool, default False
            If True, compute gradients of observables w.r.t parameters.
        Returns
        -------
        List[List[float]]
            Observables per parameter set.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        reductions = reductions or {obs: lambda x: x.mean() for obs in observables}
        self.model.eval()
        results: List[List[float]] = []

        if grad:
            self.model.train()

        with (torch.no_grad() if not grad else torch.enable_grad()):
            for i in range(0, len(parameter_sets), batch_size):
                batch_params = parameter_sets[i : i + batch_size]
                batch_tensor = torch.stack(
                    [_ensure_batch(p, self.device) for p in batch_params]
                )
                outputs = self.model(batch_tensor)
                for row in outputs:
                    row_vals: List[float] = []
                    for obs, red in zip(observables, [reductions[obs] for obs in observables]):
                        val = obs(row)
                        if isinstance(val, torch.Tensor):
                            scalar = float(red(val).cpu())
                        else:
                            scalar = float(val)
                        row_vals.append(scalar)
                    results.append(row_vals)
        return results

    def evaluate_shots(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Simulate shot noise by sampling from a normal distribution around the deterministic mean.
        """
        rng = np.random.default_rng(seed)
        raw = self.evaluate(observables, parameter_sets, batch_size=64, reductions=None)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def evaluate_grad(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        batch_size: int = 64,
    ) -> List[List[torch.Tensor]]:
        """
        Compute gradients of observables w.r.t model parameters for each parameter set.
        Returns a list of lists of gradient tensors (one per observable).
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        grads: List[List[torch.Tensor]] = []

        for values in parameter_sets:
            param_tensor = torch.tensor(
                values, dtype=torch.float32, device=self.device, requires_grad=True
            )
            output = self.model(param_tensor)
            row_grads: List[torch.Tensor] = []
            for obs in observables:
                val = obs(output)
                if isinstance(val, torch.Tensor):
                    scalar = val.mean()
                else:
                    scalar = torch.tensor(val, device=self.device, requires_grad=True)
                scalar.backward()
                grad = param_tensor.grad.clone()
                param_tensor.grad.zero_()
                row_grads.append(grad)
            grads.append(row_grads)
        return grads


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        batch_size: int = 64,
        reductions: Optional[Dict[ScalarObservable, ReductionFunction]] = None,
        grad: bool = False,
    ) -> List[List[float]]:
        raw = super().evaluate(
            observables,
            parameter_sets,
            batch_size=batch_size,
            reductions=reductions,
            grad=grad,
        )
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
