"""Enhanced estimator that supports neural‑network models, batching, shot noise, and gradient evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Protocol, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class _ModelProtocol(Protocol):
    """Protocol for objects that expose a ``forward`` method returning a tensor."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:...


class AdvancedHybridEstimator:
    """Estimator for PyTorch models with optional shot noise and gradient support.

    Parameters
    ----------
    model : _ModelProtocol
        A PyTorch module or any callable that accepts a ``torch.Tensor`` and returns a
        tensor of outputs.  The model is used in ``torch.no_grad`` mode for evaluation.
    """

    def __init__(self, model: _ModelProtocol) -> None:
        self.model = model

    @staticmethod
    def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate observables for each parameter set.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Callables that transform the model output into a scalar or tensor.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors to feed into the model.
        shots : int | None, optional
            If provided, Gaussian noise with variance ``1/shots`` is added to each
            deterministic result to simulate measurement shot noise.
        seed : int | None, optional
            Random seed for noise generation.

        Returns
        -------
        List[List[float]]
            Nested list of observable values for each parameter set.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = self._ensure_batch(params)
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

    def evaluate_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[np.ndarray]]:
        """Compute gradients of each observable w.r.t. the input parameters.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Callables that transform the model output into a scalar or tensor.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors to feed into the model.

        Returns
        -------
        List[List[np.ndarray]]
            Nested list of gradient arrays (one per observable) for each parameter set.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        grads: List[List[np.ndarray]] = []

        self.model.eval()
        for params in parameter_sets:
            inputs = self._ensure_batch(params)
            inputs.requires_grad_(True)
            outputs = self.model(inputs)

            row_grads: List[np.ndarray] = []
            for observable in observables:
                value = observable(outputs)
                if isinstance(value, torch.Tensor):
                    scalar = value.mean()
                else:
                    scalar = torch.tensor(value, dtype=torch.float32)
                scalar.backward(retain_graph=True)
                grad = inputs.grad.detach().cpu().numpy().squeeze()
                row_grads.append(grad)
                inputs.grad.zero_()

            grads.append(row_grads)

        return grads


__all__ = ["AdvancedHybridEstimator"]
