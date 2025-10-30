"""Enhanced FastBaseEstimator for PyTorch models with gradient and noise support."""

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


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables with optional gradients and noise."""

    def __init__(self, model: nn.Module, device: str | torch.device | None = None) -> None:
        self.model = model
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)
        self.model.eval()

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs.to(self.device))

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Deterministic evaluation of observables for each parameter set."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
                outputs = self._forward(inputs)
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results

    def evaluate_with_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> Tuple[List[List[float]], List[List[List[float]]]]:
        """Return observables and their gradients w.r.t parameters for each parameter set."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        obs_results: List[List[float]] = []
        grads_results: List[List[List[float]]] = []
        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device)
            inputs.requires_grad_(True)
            outputs = self._forward(inputs)
            row_obs: List[float] = []
            row_grads: List[List[float]] = []
            for obs in observables:
                value = obs(outputs)
                if isinstance(value, torch.Tensor):
                    scalar = value.mean()
                else:
                    scalar = torch.tensor(value, device=self.device, dtype=torch.float32)
                scalar.backward(retain_graph=True)
                grad = inputs.grad.clone().detach().cpu().numpy().flatten().tolist()
                row_obs.append(float(scalar))
                row_grads.append(grad)
                inputs.grad.zero_()
            obs_results.append(row_obs)
            grads_results.append(row_grads)
        return obs_results, grads_results

    def evaluate_with_shots(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Add shot noise to deterministic evaluation."""
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / np.sqrt(shots)))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator"]
