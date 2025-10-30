"""FastEstimator utilities leveraging PyTorch."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence of parameters into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastEstimator:
    """Fast deterministic and stochastic estimator for PyTorch models.

    The estimator evaluates a model on a batch of parameter sets and
    applies a list of scalar observables.  It also exposes gradient
    computation via PyTorch autograd and optional shot‑noise simulation.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Deterministic evaluation of the model.

        Parameters
        ----------
        observables:
            Callables that map the model output to a scalar.
        parameter_sets:
            Iterable of parameter sequences to evaluate.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
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

    def evaluate_with_shots(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluation that injects Gaussian shot noise.

        The noise standard deviation is ``1/sqrt(shots)``.  A seed can
        be provided for reproducibility.
        """
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def gradient(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[torch.Tensor]]:
        """Compute analytic gradients of each observable w.r.t. parameters.

        Returns a list of gradient lists for each parameter set.  Each
        inner list contains a gradient tensor for the corresponding
        observable with shape ``(num_params,)``.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        grads: List[List[torch.Tensor]] = []
        self.model.eval()
        for params in parameter_sets:
            inputs = _ensure_batch(params)
            inputs.requires_grad_(True)

            def forward() -> List[torch.Tensor]:
                outputs = self.model(inputs)
                return [obs(outputs) for obs in observables]

            obs_values = forward()
            obs_grads = []
            for val in obs_values:
                if isinstance(val, torch.Tensor):
                    grad = torch.autograd.grad(val, inputs, retain_graph=True)[0]
                else:
                    grad = torch.zeros_like(inputs)
                obs_grads.append(grad.squeeze(0))
            grads.append(obs_grads)
        return grads

__all__ = ["FastEstimator"]
