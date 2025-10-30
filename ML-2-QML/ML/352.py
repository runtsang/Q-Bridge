"""Enhanced estimator for PyTorch models with batch inference, noise injection, and gradient support."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Callable, Sequence, Optional, Tuple

# Type alias for observables that map a model output tensor to a scalar.
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence of floats to a 2‑D tensor (batch dimension)."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate PyTorch models for batches of inputs and observables, with optional Gaussian shot noise and gradient estimation."""

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Compute the mean value of each observable for every parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output tensor and returns a scalar tensor or float.
        parameter_sets : sequence of sequences
            Each inner sequence contains the input parameters for the model.
        shots : int, optional
            If provided, Gaussian noise with variance 1/shots is added to each mean.
        seed : int, optional
            Random seed for reproducibility of shot noise.
        """
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

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def compute_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[torch.Tensor]]:
        """
        Return gradients of each observable with respect to the model parameters for each parameter set.

        The gradients are flattened into a single vector per observable.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        grads: List[List[torch.Tensor]] = []

        self.model.train()
        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device)
            inputs.requires_grad = True
            outputs = self.model(inputs)

            row_grads: List[torch.Tensor] = []
            for observable in observables:
                value = observable(outputs)
                if isinstance(value, torch.Tensor):
                    loss = value.mean()
                else:
                    loss = torch.tensor(value, dtype=torch.float32, device=self.device, requires_grad=True)
                grads_wrt_params = torch.autograd.grad(
                    loss,
                    self.model.parameters(),
                    retain_graph=True,
                    allow_unused=True,
                )
                flat_grad = torch.cat(
                    [g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
                     for g, p in zip(grads_wrt_params, self.model.parameters())]
                )
                row_grads.append(flat_grad.detach().cpu())
            grads.append(row_grads)
        return grads


__all__ = ["FastBaseEstimator"]
