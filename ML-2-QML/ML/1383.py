"""FastBaseEstimator with ensemble, dropout, and caching support."""

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


class FastBaseEstimator:
    """Evaluate one or more neural networks for batches of inputs.

    The estimator accepts a single :class:`torch.nn.Module` or a list of
    modules forming an ensemble.  It supports optional Monte‑Carlo dropout
    at inference time and caches intermediate outputs for repeated
    parameter sets to avoid redundant forward passes.
    """

    def __init__(
        self,
        model: Union[nn.Module, Sequence[nn.Module]],
        *,
        device: Optional[torch.device] = None,
        cache: bool = True,
    ) -> None:
        if isinstance(model, nn.Module):
            self.models = [model]
        else:
            self.models = list(model)
        self.device = device or torch.device("cpu")
        self.cache = cache
        self._cache: dict[tuple[float,...], torch.Tensor] = {}
        for m in self.models:
            m.to(self.device)
            m.eval()

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ensemble, averaging outputs."""
        outputs = torch.stack(
            [m(inputs) for m in self.models], dim=0
        )  # shape (ensemble, batch, out)
        return outputs.mean(dim=0)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        dropout: bool = False,
        dropout_rate: float = 0.1,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """Return expectation values for each observable and parameter set.

        Parameters
        ----------
        observables
            Callables that map a model output tensor to a scalar.
        parameter_sets
            Iterable of parameter vectors to evaluate.
        dropout
            If True, enable dropout layers during inference.
        dropout_rate
            Dropout probability used when ``dropout=True``.
        seed
            Random seed for reproducible dropout masks.
        """
        if not observables:
            observables = [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        if seed is not None:
            torch.manual_seed(seed)

        for params in parameter_sets:
            key = tuple(params)
            if self.cache and key in self._cache:
                outputs = self._cache[key]
            else:
                inputs = _ensure_batch(params).to(self.device)
                if dropout:
                    for m in self.models:
                        for module in m.modules():
                            if isinstance(module, nn.Dropout):
                                module.p = dropout_rate
                outputs = self._forward(inputs)
                if self.cache:
                    self._cache[key] = outputs

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

    def __repr__(self) -> str:
        return f"<FastBaseEstimator models={len(self.models)} device={self.device!s}>"

__all__ = ["FastBaseEstimator"]
