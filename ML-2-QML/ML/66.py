"""Enhanced estimator utilities built on PyTorch.

This module extends the original lightweight estimator with GPU support,
caching, batch processing, and a simple training API.  The public
interface remains compatible with the seed, while the added features
enable more realistic experimentation.
"""

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


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables.

    The estimator is device‑aware, supports optional result caching,
    and exposes a minimal training helper.
    """

    def __init__(self, model: nn.Module, device: str | torch.device | None = None, cache: bool = False) -> None:
        self.model = model
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        self.cache = cache
        self._cache: dict[tuple[float,...], List[float]] = {}

    def _run(self, inputs: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(inputs.to(self.device))

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Return a list of observable values for each parameter set.

        Parameters
        ----------
        observables:
            Callables that map the model output to a scalar.
        parameter_sets:
            Sequence of parameter vectors.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        for params in parameter_sets:
            key = tuple(params)
            if self.cache and key in self._cache:
                results.append(self._cache[key])
                continue

            inputs = _ensure_batch(params).to(self.device)
            outputs = self._run(inputs)
            row: List[float] = []

            for observable in observables:
                value = observable(outputs)
                if isinstance(value, torch.Tensor):
                    scalar = float(value.mean().cpu())
                else:
                    scalar = float(value)
                row.append(scalar)

            if self.cache:
                self._cache[key] = row
            results.append(row)

        return results

    def predict(self, parameter_sets: Sequence[Sequence[float]]) -> torch.Tensor:
        """Return raw model outputs for the supplied parameters."""
        self.model.eval()
        with torch.no_grad():
            inputs = torch.cat([_ensure_batch(p) for p in parameter_sets], dim=0).to(self.device)
            return self.model(inputs)

    def evaluate_batch(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        batch_size: int = 256,
    ) -> List[List[float]]:
        """Evaluate in mini‑batches to keep memory usage bounded."""
        results: List[List[float]] = []
        for start in range(0, len(parameter_sets), batch_size):
            batch = parameter_sets[start : start + batch_size]
            results.extend(self.evaluate(observables, batch))
        return results

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        epochs: int = 10,
        batch_size: int = 32,
        device: str | torch.device | None = None,
    ) -> None:
        """Very small training loop for quick prototyping."""
        device = device or self.device
        self.model.train()
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()


class FastEstimator(FastBaseEstimator):
    """Add optional Gaussian shot noise to the deterministic estimator."""

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
