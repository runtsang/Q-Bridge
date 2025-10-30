"""Hybrid estimator for classical neural networks with noise and gradient support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastHybridEstimator:
    """Evaluate a PyTorch model for a set of inputs and observable functions.

    The estimator supports:
    * Batched parameter evaluation.
    * Optional Gaussian shot noise to mimic measurement statistics.
    * Gradient extraction via autograd for each observable.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Parameters
        ----------
        observables
            Iterable of functions mapping model outputs to scalar values.
        parameter_sets
            Sequence of parameter vectors to feed to the model.
        shots
            If provided, adds Gaussian noise with std=1/sqrt(shots).
        seed
            Random seed for reproducibility of the noise.

        Returns
        -------
        List of [batch][observable] values.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []

                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    row.append(val)

                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy = []
        for row in results:
            noisy.append([float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row])

        return noisy

    def evaluate_with_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> Tuple[List[List[float]], List[List[List[float]]]]:
        """
        Compute both the forward values and the gradients of each observable
        with respect to the input parameters.

        Returns
        -------
        results
            List of [batch][observable] output values.
        gradients
            List of [batch][observable][param] gradients.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        gradients: List[List[List[float]]] = []

        self.model.train()
        for params in parameter_sets:
            inputs = _ensure_batch(params)
            inputs.requires_grad_(True)
            outputs = self.model(inputs)

            row: List[float] = []
            grad_row: List[List[float]] = []

            for obs in observables:
                val = obs(outputs)
                if isinstance(val, torch.Tensor):
                    val = val.mean()
                row.append(val.item())

                val.backward(retain_graph=True)
                grad = inputs.grad.clone().detach()
                grad_row.append(grad.squeeze().tolist())
                inputs.grad.zero_()

            results.append(row)
            gradients.append(grad_row)

        self.model.eval()
        return results, gradients


__all__ = ["FastHybridEstimator"]
