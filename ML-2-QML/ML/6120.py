"""Enhanced FastBaseEstimator using PyTorch with GPU support and gradient capabilities."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator(nn.Module):
    """Neural network estimator that evaluates observables and gradients for batches of parameters."""

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

    def _forward(self, params: torch.Tensor) -> torch.Tensor:
        return self.model(params)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        return_gradients: bool = False,
        shot_noise: int | None = None,
        rng_seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate observables (and optionally gradients) for each parameter set.

        Parameters
        ----------
        observables : iterable of callables that map model outputs to scalars.
            If None, a default mean is used.
        parameter_sets : list of parameter sequences.
        return_gradients : if True, also return gradients of each observable w.r.t parameters.
        shot_noise : if provided, add Gaussian noise with variance 1/shots.
        rng_seed : seed for noise generation.

        Returns
        -------
        List of rows; each row contains observable values (and optionally gradients flattened).
        """
        if observables is None:
            observables = [lambda out: out.mean(dim=-1)]
        observables = list(observables)
        if parameter_sets is None:
            raise ValueError("parameter_sets must be provided")
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                batch = _ensure_batch(params).to(self.device)
                outputs = self._forward(batch)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)

        if shot_noise is not None:
            rng = np.random.default_rng(rng_seed)
            noisy_results = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shot_noise))) for mean in row]
                noisy_results.append(noisy_row)
            results = noisy_results

        if return_gradients:
            grads = self._compute_gradients(observables, parameter_sets)
            results = [row + grad for row, grad in zip(results, grads)]

        return results

    def _compute_gradients(
        self,
        observables: list[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Compute gradients of each observable w.r.t parameters for each parameter set.
        Returns flattened gradient lists.
        """
        grads_list: List[List[float]] = []
        self.model.train()
        for params in parameter_sets:
            batch = _ensure_batch(params).to(self.device)
            batch.requires_grad_(True)
            outputs = self._forward(batch)
            grads_row: List[float] = []
            for obs in observables:
                val = obs(outputs)
                if isinstance(val, torch.Tensor):
                    val = val.mean()
                else:
                    val = torch.tensor(val, device=self.device)
                val.backward(retain_graph=True)
                grads = batch.grad.squeeze().cpu().numpy()
                grads_row.extend(grads.tolist())
                batch.grad.zero_()
            grads_list.append(grads_row)
        return grads_list


__all__ = ["FastBaseEstimator"]
