"""Enhanced estimator utilities built on PyTorch with GPU and DataLoader support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float], device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class Gen200BaseEstimator:
    """Evaluate neural networks over batches of parameters with optional GPU support.

    Parameters
    ----------
    model
        PyTorch ``nn.Module`` that maps a parameter vector to a feature vector.
    device
        Torch device to run the model on; defaults to ``cpu``.
    batch_size
        Size of internal DataLoader; controls memory consumption for large parameter sets.
    """

    def __init__(self, model: nn.Module, device: torch.device | str = "cpu", batch_size: int = 256) -> None:
        self.model = model
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model.to(self.device)

    def _evaluate_raw(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        inputs_tensor = _ensure_batch(np.array(parameter_sets, dtype=np.float32), self.device)
        dataset = TensorDataset(inputs_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0]
                outputs = self.model(inputs)
                for idx in range(inputs.shape[0]):
                    row: List[float] = []
                    for observable in observables:
                        value = observable(outputs[idx : idx + 1])
                        if isinstance(value, torch.Tensor):
                            scalar = float(value.mean().cpu())
                        else:
                            scalar = float(value)
                        row.append(scalar)
                    results.append(row)
        return results

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute scalar outputs for each parameter set."""
        return self._evaluate_raw(observables, parameter_sets)


class Gen200Estimator(Gen200BaseEstimator):
    """Adds configurable shot‑noise to the deterministic estimator."""

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


__all__ = ["Gen200BaseEstimator", "Gen200Estimator"]
