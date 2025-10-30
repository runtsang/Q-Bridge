"""Enhanced FastBaseEstimator for PyTorch models with batched evaluation, optional dropout, and gradient support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

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
    """Evaluate PyTorch models for batches of inputs and observables with optional shot noise and gradient computation."""

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        return_gradients: bool = False,
        dropout: bool = False,
    ) -> List[List[float]]:
        """Compute observables for each parameter set.

        Args:
            observables: Callables that map model outputs to scalars.
            parameter_sets: Iterable of parameter vectors.
            shots: If provided, adds Gaussian shot noise with variance 1/shots.
            seed: RNG seed for reproducibility.
            return_gradients: If True, returns gradients of observables w.r.t. inputs.
            dropout: If True, enables dropout layers during evaluation.

        Returns:
            A list of rows, each containing the observable values (and optionally gradients) for a parameter set.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        if dropout:
            # Enable dropout during evaluation
            def _enable_dropout(m: nn.Module) -> None:
                if isinstance(m, nn.Dropout):
                    m.train()
            self.model.apply(_enable_dropout)

        rng = np.random.default_rng(seed)

        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
                inputs.requires_grad_(True)
                outputs = self.model(inputs)
                row: List[float] = []

                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    if shots is not None:
                        scalar += rng.normal(0, max(1e-6, 1 / shots))
                    row.append(scalar)

                if return_gradients:
                    grads: List[float] = []
                    for observable in observables:
                        self.model.zero_grad()
                        value = observable(outputs)
                        if isinstance(value, torch.Tensor):
                            value = value.mean()
                        value.backward(retain_graph=True)
                        grad = inputs.grad
                        grads.append(float(grad.abs().mean().item()))
                    row.extend(grads)

                results.append(row)

        return results


__all__ = ["FastBaseEstimator"]
