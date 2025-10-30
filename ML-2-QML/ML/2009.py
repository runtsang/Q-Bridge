"""Enhanced estimator utilities leveraging PyTorch for efficient batch and gradient evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[Sequence[float]]) -> torch.Tensor:
    """Convert a list of parameter vectors into a 2‑D tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastEstimatorGen210:
    """
    A lightweight estimator that evaluates a neural network on many parameter sets.
    Supports deterministic evaluation, optional shot‑noise simulation, and gradient
    computation via autograd.  Designed to run on CPU or GPU.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None) -> None:
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Compute observables for each parameter set deterministically.
        """
        observables = list(observables) or [lambda output: output.mean(dim=-1)]
        param_tensor = _ensure_batch(parameter_sets).to(self.device)
        with torch.no_grad():
            outputs = self.model(param_tensor)
        results: List[List[float]] = []
        for out in outputs:
            row: List[float] = []
            for obs in observables:
                val = obs(out)
                if isinstance(val, torch.Tensor):
                    scalar = float(val.mean().cpu())
                else:
                    scalar = float(val)
                row.append(scalar)
            results.append(row)
        return results

    def evaluate_with_shots(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Same as :py:meth:`evaluate` but adds Gaussian shot noise to each mean.
        """
        base = self.evaluate(observables, parameter_sets)
        if shots is None:
            return base
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in base:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy

    def evaluate_with_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[List[float]]]:
        """
        Return gradients of each observable w.r.t. the parameters.
        The outer list is over parameter sets, the middle list over observables,
        and the innermost list contains the gradient vector.
        """
        observables = list(observables) or [lambda output: output.mean(dim=-1)]
        grads: List[List[List[float]]] = []

        for params in parameter_sets:
            params_tensor = torch.tensor(params, dtype=torch.float32, device=self.device, requires_grad=True)
            output = self.model(params_tensor)
            row_grads: List[List[float]] = []
            for obs in observables:
                val = obs(output)
                if isinstance(val, torch.Tensor):
                    mean_val = val.mean()
                else:
                    mean_val = torch.tensor(val, device=self.device)
                mean_val.backward(retain_graph=True)
                grad_vec = params_tensor.grad.detach().cpu().numpy().tolist()
                row_grads.append(grad_vec)
                params_tensor.grad.zero_()
            grads.append(row_grads)
        return grads
