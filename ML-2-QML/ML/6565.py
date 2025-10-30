"""Enhanced estimator utilities leveraging PyTorch's autograd and GPU acceleration."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Tuple

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
GradientObservable = Callable[[torch.Tensor], torch.Tensor]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """General-purpose estimator for deterministic neural networks.

    Supports batched inference, optional GPU execution, and analytical gradients
    of supplied scalar observables via PyTorch autograd.
    """
    def __init__(self, model: nn.Module, device: torch.device | str = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        compute_gradients: bool = False,
        gradient_observables: Optional[Iterable[GradientObservable]] = None,
    ) -> Tuple[List[List[float]], Optional[List[List[torch.Tensor]]]]:
        """Return scalar observables and optionally their gradients.

        Parameters
        ----------
        observables
            Scalar observable functions applied to the model output.
        parameter_sets
            Iterable of parameter vectors.
        compute_gradients
            If True, compute gradients of each observable w.r.t. input parameters.
        gradient_observables
            Functions that map the model output to a tensor suitable for
            gradient computation. If None and gradients are requested,
            the same observables are used.
        """
        observables = list(observables) or [lambda x: x.mean(dim=-1)]
        if compute_gradients and gradient_observables is None:
            gradient_observables = observables

        results: List[List[float]] = []
        grads: List[List[torch.Tensor]] | None = None

        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
                outputs = self.model(inputs)

                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)

        if compute_gradients:
            grads = []
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
                inputs.requires_grad_(True)
                outputs = self.model(inputs)

                grad_row: List[torch.Tensor] = []
                for grad_obs in gradient_observables:
                    val = grad_obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar_val = val.sum()
                    else:
                        scalar_val = torch.tensor(val, dtype=torch.float32, device=self.device)
                    scalar_val.backward(retain_graph=True)
                    grad = inputs.grad.clone().detach().cpu()
                    grad_row.append(grad)
                    inputs.grad.zero_()
                grads.append(grad_row)

        return results, grads

class FastEstimator(FastBaseEstimator):
    """Adds optional stochastic shot noise to deterministic predictions.

    Parameters
    ----------
    shots
        Number of measurement shots. If None, no noise is added.
    seed
        Random seed for reproducibility.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str = "cpu",
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(model, device)
        self.shots = shots
        self.rng = np.random.default_rng(seed)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        compute_gradients: bool = False,
        gradient_observables: Optional[Iterable[GradientObservable]] = None,
    ) -> Tuple[List[List[float]], Optional[List[List[torch.Tensor]]]]:
        raw_results, raw_grads = super().evaluate(
            observables,
            parameter_sets,
            compute_gradients=compute_gradients,
            gradient_observables=gradient_observables,
        )
        if self.shots is None:
            return raw_results, raw_grads

        noisy_results = []
        for row in raw_results:
            noisy_row = [float(self.rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
            noisy_results.append(noisy_row)

        return noisy_results, raw_grads

__all__ = ["FastBaseEstimator", "FastEstimator"]
