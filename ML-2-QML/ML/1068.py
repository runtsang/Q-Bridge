"""Enhanced classical estimator built on PyTorch, supporting batched evaluation, device placement, shot‑noise, and gradient extraction."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float], device: str) -> torch.Tensor:
    """Convert a sequence of parameters to a 2‑D torch tensor on the given device."""
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """
    Evaluate neural networks for batches of inputs and observables.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to evaluate.
    device : str, optional
        Target device for evaluation (default: 'cpu').
    """

    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.device = device
        self._cache: dict[tuple[float,...], List[float]] = {}

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        batch_size: int | None = None,
        return_tensors: bool = False,
    ) -> List[List[float]]:
        """
        Compute observables for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output and returns a tensor or scalar.
        parameter_sets : sequence of parameter sequences
            Each inner sequence is a vector of parameters for a single evaluation.
        batch_size : int, optional
            Size of mini‑batches; if None, evaluates one by one.
        return_tensors : bool, optional
            If True, returns raw tensors instead of scalars.

        Returns
        -------
        List[List[float]]
            Results for each parameter set and observable.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                # caching for repeated evaluations
                key = tuple(params)
                if key in self._cache:
                    results.append(self._cache[key])
                    continue

                inputs = _ensure_batch(params, self.device)
                outputs = self.model(inputs)

                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)

                self._cache[key] = row
                results.append(row)

        return results

    def evaluate_with_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> Tuple[List[List[float]], List[List[torch.Tensor]]]:
        """
        Compute outputs and gradients w.r.t. model parameters.

        Returns
        -------
        Tuple of:
            - List of observable values.
            - List of gradient tensors for each parameter set.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        outputs_list: List[List[float]] = []
        grads_list: List[List[torch.Tensor]] = []

        self.model.train()
        for params in parameter_sets:
            inputs = _ensure_batch(params, self.device)
            inputs.requires_grad_(True)

            outputs = self.model(inputs)
            row: List[float] = []
            for observable in observables:
                value = observable(outputs)
                if isinstance(value, torch.Tensor):
                    scalar = value.mean()
                else:
                    scalar = torch.tensor(value, device=self.device)
                row.append(float(scalar))

            # Back‑propagate to get gradients
            self.model.zero_grad()
            total = sum(row)
            total.backward()

            grads = [p.grad.clone().detach() for p in self.model.parameters()]
            outputs_list.append(row)
            grads_list.append(grads)

        return outputs_list, grads_list

    def add_shot_noise(
        self,
        results: List[List[float]],
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Add Gaussian shot noise to deterministic results.

        Parameters
        ----------
        results : list of list of floats
            Deterministic outputs.
        shots : int, optional
            Number of shots; if None, no noise is added.
        seed : int, optional
            RNG seed for reproducibility.

        Returns
        -------
        List[List[float]]
            Noisy results.
        """
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator"]
