"""Enhanced estimator for PyTorch models with batched evaluation and gradient support."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
GradObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for multiple parameter sets and observables.

    Supports batched inference, optional GPU execution, and gradient
    computation via autograd.  Observables can be arbitrary differentiable
    functions of the model output.
    """

    def __init__(self, model: nn.Module, device: torch.device | str = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a list of observable values for each parameter set."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
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
        *,
        retain_graph: bool = False,
    ) -> List[List[Tuple[torch.Tensor,...]]]:
        """Compute gradients of each observable w.r.t. the model parameters.

        Returns a list of tuples, one per observable, containing the gradient
        tensors for each parameter in the model.  The gradients are detached
        and moved to CPU.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        grads_list: List[List[Tuple[torch.Tensor,...]]] = []

        self.model.train()
        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device).requires_grad_(True)
            outputs = self.model(inputs)
            row: List[Tuple[torch.Tensor,...]] = []
            for observable in observables:
                value = observable(outputs)
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, dtype=outputs.dtype, device=outputs.device)
                loss = value.sum()
                grads = torch.autograd.grad(loss, self.model.parameters(), retain_graph=retain_graph)
                grads_detached = tuple(g.detach().cpu() if g is not None else torch.zeros_like(p.cpu())
                                       for g, p in zip(grads, self.model.parameters()))
                row.append(grads_detached)
            grads_list.append(row)
        return grads_list


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator.

    The noise model is a simple normal distribution with variance 1/shots.
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
