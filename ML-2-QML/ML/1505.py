"""Enhanced estimator utilities with GPU support, batched evaluation, and gradient estimation."""

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


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables.

    The model is moved to the best available device (GPU if present) and
    evaluation can be performed in batches.  Observables are callables
    that accept the model output tensor and return either a scalar tensor
    or a python float.
    """

    def __init__(self, model: nn.Module, device: torch.device | str | None = None) -> None:
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        batch_size: int | None = None,
    ) -> List[List[float]]:
        """Return a list of observable values for each parameter set.

        Parameters
        ----------
        observables:
            Callables that convert the model output to a scalar.
        parameter_sets:
            Iterable of parameter tuples, each matched to the model's input shape.
        batch_size:
            Optional batch size; if None all parameters are processed at once.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            if batch_size is None:
                batch_size = len(parameter_sets)
            for i in range(0, len(parameter_sets), batch_size):
                batch = parameter_sets[i : i + batch_size]
                inputs = _ensure_batch(batch).to(self.device)
                outputs = self.model(inputs)
                for outputs_row in outputs:
                    row: List[float] = []
                    for observable in observables:
                        value = observable(outputs_row)
                        if isinstance(value, torch.Tensor):
                            scalar = float(value.mean().cpu())
                        else:
                            scalar = float(value)
                        row.append(scalar)
                    results.append(row)
        return results

    def predict(self, parameter_sets: Sequence[Sequence[float]]) -> torch.Tensor:
        """Return raw model outputs for the provided parameters."""
        self.model.eval()
        with torch.no_grad():
            inputs = _ensure_batch(parameter_sets).to(self.device)
            return self.model(inputs).cpu()

    def gradient(
        self,
        observables: Iterable[ScalarObservable],
        parameter_set: Sequence[float],
        param_index: int,
    ) -> float:
        """Compute the gradient of the first observable w.r.t a single input
        parameter using autograd.  Only the first observable is returned."""
        if not isinstance(param_index, int):
            raise TypeError("param_index must be an integer")
        self.model.train()
        inputs = _ensure_batch([parameter_set]).requires_grad_(True).to(self.device)
        outputs = self.model(inputs)
        observable = list(observables)[0]
        value = observable(outputs[0])
        if isinstance(value, torch.Tensor):
            value = value.mean()
        value.backward()
        grad = inputs.grad[0, param_index].item()
        return grad


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator
    and optional GPU training support."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        batch_size: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets, batch_size=batch_size)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def train(
        self,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        epochs: int = 1,
        device: torch.device | str | None = None,
    ) -> None:
        """Simple training loop for the underlying model.

        Parameters
        ----------
        loss_fn:
            Loss function that takes (outputs, targets).
        optimizer:
            Optimizer to update model parameters.
        data_loader:
            Iterable of (inputs, targets) tuples.
        epochs:
            Number of training epochs.
        device:
            Optional device override; defaults to the estimator's device.
        """
        if device is None:
            device = self.device
        self.model.train()
        for _ in range(epochs):
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
