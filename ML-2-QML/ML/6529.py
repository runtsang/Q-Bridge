"""Enhanced FastBaseEstimator utilities for PyTorch models with GPU, caching, and gradient support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of values to a 2‑D float tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for batches of parameter sets and observables."""

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None) -> None:
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(inputs.to(self.device))

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Deterministic evaluation of observables for each parameter set."""
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
        parameter_sets: torch.Tensor,
        batch_size: int = 256,
    ) -> List[List[float]]:
        """Vectorised evaluation over a large batch of parameters."""
        self.model.eval()
        results: List[List[float]] = []
        loader = DataLoader(parameter_sets, batch_size=batch_size, shuffle=False)
        for batch in loader:
            batch = batch.to(self.device)
            outputs = self.model(batch)
            for obs in observables:
                val = obs(outputs)
                if isinstance(val, torch.Tensor):
                    val = val.cpu().numpy()
                else:
                    val = np.array(val)
                results.append(val)
        # Transpose to match original shape
        return [list(row) for row in zip(*results)]

    def evaluate_with_shots(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Return noisy estimates by adding Gaussian noise to deterministic outputs."""
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def evaluate_with_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[Tuple[float, torch.Tensor]]]:
        """
        Compute gradients of each observable w.r.t. each parameter.
        Returns a list of rows, each containing tuples of (observable_value, gradient_tensor).
        """
        self.model.train()
        results: List[List[Tuple[float, torch.Tensor]]] = []
        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device)
            inputs.requires_grad_(True)
            outputs = self.model(inputs)
            row: List[Tuple[float, torch.Tensor]] = []
            for observable in observables:
                value = observable(outputs)
                if isinstance(value, torch.Tensor):
                    value = value.mean()
                else:
                    value = torch.tensor(value, device=self.device)
                grad = torch.autograd.grad(value, inputs, retain_graph=True, create_graph=False)[0]
                row.append((float(value.cpu()), grad.cpu()))
            results.append(row)
        return results

    def cache_parameters(self, parameter_sets: Sequence[Sequence[float]]) -> None:
        """Cache parameter sets for quick repeated evaluation."""
        self._cached_params = _ensure_batch(parameter_sets).to(self.device)


__all__ = ["FastBaseEstimator"]
