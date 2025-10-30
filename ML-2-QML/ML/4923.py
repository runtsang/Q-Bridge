"""Combined fast estimator that unifies classical PyTorch models and quantum-inspired modules."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastCombinedEstimator:
    """
    Evaluate a set of models (PyTorch nn.Module or callable) on parameter sets.
    Supports deterministic evaluation and optional Gaussian shot noise.
    """

    def __init__(self, models: Sequence[Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]]) -> None:
        """
        :param models: sequence of PyTorch modules or callables that accept a 2‑D tensor
                       of parameters and return a tensor of outputs.
        """
        self.models = list(models)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[List[float]]]:
        """
        Return a 3‑D list: outer dimension over models,
        middle over parameter sets, inner over observables.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        rng = np.random.default_rng(seed)

        results: List[List[List[float]]] = []

        for model in self.models:
            raw_rows: List[List[float]] = []
            if isinstance(model, nn.Module):
                model.eval()
            with torch.no_grad():
                for params in parameter_sets:
                    inputs = _ensure_batch(params)
                    outputs = model(inputs) if isinstance(model, nn.Module) else model(inputs)
                    row: List[float] = []
                    for observable in observables:
                        value = observable(outputs)
                        scalar = float(value.mean().cpu()) if isinstance(value, torch.Tensor) else float(value)
                        row.append(scalar)
                    raw_rows.append(row)

            if shots is None:
                results.append(raw_rows)
                continue

            noisy_rows: List[List[float]] = []
            for row in raw_rows:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy_rows.append(noisy_row)
            results.append(noisy_rows)

        return results


__all__ = ["FastCombinedEstimator"]
