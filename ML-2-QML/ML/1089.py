"""Enhanced lightweight estimator utilities implemented with PyTorch modules."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence of floats to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate one or more neural networks for batches of inputs and observables.

    Parameters
    ----------
    models : nn.Module | Sequence[nn.Module]
        A single model or a list of models to evaluate.  All models are moved to the
        default device (CUDA if available, otherwise CPU).
    """

    def __init__(self, models: nn.Module | Sequence[nn.Module]) -> None:
        if isinstance(models, nn.Module):
            self.models = [models]
        else:
            self.models = list(models)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for model in self.models:
            model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        device: Optional[torch.device] = None,
        dropout: bool = False,
        dropout_prob: float = 0.1,
    ) -> List[List[List[float]]]:
        """
        Compute observables for each model and parameter set.

        Returns a list of shape [num_models, num_params, num_observables].
        """
        if device is None:
            device = self.device
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[List[float]]] = []

        for model in self.models:
            # Toggle dropout if requested
            if dropout:
                model.train()
                for m in model.modules():
                    if isinstance(m, nn.Dropout):
                        m.p = dropout_prob
            else:
                model.eval()

            with torch.no_grad():
                model_results: List[List[float]] = []
                for params in parameter_sets:
                    inputs = _ensure_batch(params).to(device)
                    outputs = model(inputs)
                    row: List[float] = []
                    for observable in observables:
                        value = observable(outputs)
                        if isinstance(value, torch.Tensor):
                            scalar = float(value.mean().cpu())
                        else:
                            scalar = float(value)
                        row.append(scalar)
                    model_results.append(row)
                results.append(model_results)

        return results


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        device: Optional[torch.device] = None,
        dropout: bool = False,
        dropout_prob: float = 0.1,
    ) -> List[List[List[float]]]:
        raw = super().evaluate(
            observables,
            parameter_sets,
            device=device,
            dropout=dropout,
            dropout_prob=dropout_prob,
        )
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[List[float]]] = []
        for model_res in raw:
            noisy_model: List[List[float]] = []
            for row in model_res:
                noisy_row = [
                    float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
                ]
                noisy_model.append(noisy_row)
            noisy.append(noisy_model)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
