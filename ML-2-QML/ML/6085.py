"""Enhanced estimator utilities implemented with PyTorch modules.

The estimator supports batched evaluation, GPU acceleration, optional Gaussian shot noise,
and optional gradient computation via PyTorch autograd. It is a drop‑in replacement
for the original FastEstimator while providing richer functionality for research.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
GradObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastEstimator:
    """Evaluate neural networks for batches of inputs and observables with optional gradient support.

    The estimator operates in evaluation mode and can optionally compute gradients of the
    observables w.r.t. the model parameters using PyTorch's autograd. It also supports
    GPU execution when a CUDA device is available.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None) -> None:
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        compute_gradients: bool = False,
        grad_observables: Optional[Iterable[GradObservable]] = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate the model on a collection of parameter sets.

        Parameters
        ----------
        observables : iterable of callables
            Functions that accept the model output tensor and return a scalar (tensor or float).
        parameter_sets : sequence of sequences
            Each inner sequence contains the scalar parameters to feed into the model.
        compute_gradients : bool
            If True, compute gradients of the provided grad_observables.
        grad_observables : iterable of callables
            Functions that accept the model output tensor and return a scalar tensor.
            Required if compute_gradients is True.
        shots : int, optional
            When provided, Gaussian shot noise with variance 1/shots is added to the results.
        seed : int, optional
            Seed for the random number generator used when adding shot noise.

        Returns
        -------
        results : list of lists
            Each inner list contains the evaluated observables for one parameter set.
            If gradients are requested, the corresponding gradient values are appended after the observables.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        if compute_gradients:
            if grad_observables is None:
                raise ValueError("grad_observables must be provided when compute_gradients=True")
            grad_observables = list(grad_observables)
        else:
            grad_observables = []

        rng = np.random.default_rng(seed)
        results: List[List[float]] = []

        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
                outputs = self.model(inputs)

                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    scalar = float(value.mean().cpu())
                    row.append(scalar)

                if shots is not None:
                    row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]

                if compute_gradients:
                    # Enable gradient tracking for the outputs
                    outputs.requires_grad_(True)
                    for grad_observable in grad_observables:
                        grad_value = grad_observable(outputs)
                        grad_value.backward()
                        # Aggregate gradient norm across all parameters
                        grad_norm = sum(p.grad.norm().item() for p in self.model.parameters() if p.grad is not None)
                        row.append(float(grad_norm))
                        outputs.grad.zero_()

                results.append(row)

        return results


__all__ = ["FastEstimator"]
