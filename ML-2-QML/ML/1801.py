"""Advanced FastBaseEstimator for PyTorch with gradient support and shot noise."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Callable, Iterable, List, Sequence, Tuple, Union

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables with optional gradient computation."""

    def __init__(self, model: nn.Module, device: str | torch.device | None = None) -> None:
        self.device = torch.device(device or "cpu")
        self.model = model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Deterministic evaluation of a list of observables over a batch of parameter sets."""
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
        return results

    def evaluate_batch(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Vectorized evaluation that returns a NumPy array of shape (batch, observables)."""
        results = self.evaluate(observables, parameter_sets)
        return np.asarray(results, dtype=np.float64)

    def evaluate_with_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute both expectation values and gradients w.r.t. all parameters.

        Returns:
            results: shape (n_samples, n_observables)
            grads:   shape (n_samples, n_params, n_observables)
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        n_samples = len(parameter_sets)
        n_observables = len(observables)

        params_tensor = torch.as_tensor(parameter_sets, dtype=torch.float32, device=self.device)
        params_tensor.requires_grad_(True)

        with torch.no_grad():
            outputs = self.model(params_tensor)

        # Allocate containers
        results = np.empty((n_samples, n_observables), dtype=np.float64)
        grads = np.empty((n_samples, params_tensor.shape[1], n_observables), dtype=np.float64)

        for j, observable in enumerate(observables):
            # Forward pass for this observable
            if isinstance(observable, nn.Module):
                values = observable(outputs).squeeze(-1)
            else:
                values = observable(outputs).squeeze(-1)

            # Compute gradients for each sample
            grad_values = torch.autograd.grad(
                outputs=values,
                inputs=params_tensor,
                grad_outputs=torch.ones_like(values),
                retain_graph=True,
                allow_unused=True,
            )[0]

            # Store results and gradients
            results[:, j] = values.cpu().numpy()
            grads[:, :, j] = grad_values.cpu().numpy()

        return results, grads


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
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
