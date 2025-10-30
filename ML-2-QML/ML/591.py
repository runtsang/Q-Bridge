"""Lightweight estimator utilities implemented with PyTorch modules.

Extended to support device placement, batched evaluation, custom observables,
and gradient computation for hybrid workflows."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[Sequence[float]]) -> torch.Tensor:
    """Convert a sequence of parameter vectors into a 2‑D tensor."""
    arr = np.array(values, dtype=np.float32)
    return torch.from_numpy(arr)


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables."""

    def __init__(self, model: nn.Module, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a list of observable values for each parameter set."""
        obs_list = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            batch = _ensure_batch(parameter_sets).to(self.device)
            outputs = self.model(batch)
            for out in outputs:
                row: List[float] = []
                for obs in obs_list:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

    def evaluate_batch(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> torch.Tensor:
        """Return a dense tensor of shape (batch, num_observables)."""
        obs_list = list(observables) or [lambda out: out.mean(dim=-1)]
        batch = _ensure_batch(parameter_sets).to(self.device)
        outputs = self.model(batch)
        batch_results: List[List[float]] = []
        for out in outputs:
            row = [float(obs(out).mean().cpu()) if isinstance(obs(out), torch.Tensor) else float(obs(out)) for obs in obs_list]
            batch_results.append(row)
        return torch.tensor(batch_results, device=self.device)

    def evaluate_and_grad(
        self,
        observable: ScalarObservable,
        parameter_sets: Sequence[Sequence[float]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the observable and its gradient w.r.t. model parameters for
        each parameter set. Returns (values, gradients).
        """
        values = torch.empty(len(parameter_sets), device=self.device)
        grads = torch.empty((len(parameter_sets), *self.model.parameters().__len__()), device=self.device)
        for i, params in enumerate(parameter_sets):
            batch = torch.tensor(params, dtype=torch.float32, device=self.device).unsqueeze(0)
            batch.requires_grad_(True)
            out = self.model(batch)
            val = observable(out)
            values[i] = val.mean()
            grads[i] = torch.autograd.grad(val, self.model.parameters(), retain_graph=True)[0].flatten()
        return values, grads


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
