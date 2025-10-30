"""Enhanced estimator utilities with GPU support, batched evaluation, and gradient computation."""

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
    """Base class to evaluate a PyTorch model for a set of input parameter sets."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(
                    self.model.device if hasattr(self.model, "device") else "cpu"
                )
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


class FastEstimator(FastBaseEstimator):
    """Extension of FastBaseEstimator with GPU support, batched evaluation,
    optional shot noise, and gradient computation via autograd."""

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        super().__init__(model)
        self.device = torch.device(device)
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate the model and optionally add Gaussian shot noise."""
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def evaluate_and_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> Tuple[List[List[float]], List[List[List[np.ndarray]]]]:
        """
        Evaluate the model and compute gradients of each observable with respect to
        the input parameters using PyTorch's autograd.
        Returns a tuple (results, gradients) where gradients is a list of rows,
        each containing a list of gradient arrays (one per observable).
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        gradients: List[List[List[np.ndarray]]] = []

        self.model.eval()
        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device)
            inputs.requires_grad_(True)

            outputs = self.model(inputs)

            row_vals: List[float] = []
            row_grads: List[List[np.ndarray]] = []

            for observable in observables:
                value = observable(outputs)
                if isinstance(value, torch.Tensor):
                    scalar = value.mean()
                else:
                    scalar = torch.tensor(value, dtype=torch.float32, device=self.device)
                row_vals.append(float(scalar.cpu()))

                scalar.backward(retain_graph=True)
                grad_np = inputs.grad.detach().cpu().numpy()
                row_grads.append(grad_np)
                inputs.grad.zero_()

            results.append(row_vals)
            gradients.append(row_grads)

        return results, gradients


__all__ = ["FastBaseEstimator", "FastEstimator"]
