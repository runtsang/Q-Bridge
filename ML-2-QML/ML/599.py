"""Enhanced classical estimator with GPU support, dropout, caching, and autograd."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import Dropout

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """A lightweight, GPU‑aware estimator for neural network models.

    Parameters
    ----------
    model: nn.Module
        The PyTorch model to evaluate.
    device: str | torch.device | None, default=None
        Target device. If None, use ``torch.device('cuda' if torch.cuda.is_available() else 'cpu')``.
    dropout: float | None, default=None
        If provided, add a dropout layer after each hidden layer to regularise predictions.
    cache: bool, default=False
        Enable caching of intermediate outputs for repeated parameter sets.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        device: Optional[torch.device | str] = None,
        dropout: Optional[float] = None,
        cache: bool = False,
    ) -> None:
        self.model = model
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.dropout = dropout
        self.cache: Dict[Tuple[float,...], torch.Tensor] = {} if cache else None

        if dropout is not None:
            # Insert dropout layers after each linear module
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    setattr(self.model, name + "_dropout", Dropout(dropout))

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.cache is not None:
            key = tuple(inputs.squeeze().tolist())
            if key in self.cache:
                return self.cache[key]
        with torch.no_grad():
            outputs = self.model(inputs.to(self.device))
        if self.cache is not None:
            self.cache[key] = outputs.cpu()
        return outputs.cpu()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[Dict[str, float]]:
        """Evaluate observables for each parameter set.

        Returns
        -------
        List[Dict[str, float]]
            A list of dictionaries mapping observable names to scalar values.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[Dict[str, float]] = []
        for params in parameter_sets:
            inputs = _ensure_batch(params)
            outputs = self._forward(inputs)
            row: Dict[str, float] = {}
            for i, observable in enumerate(observables):
                value = observable(outputs)
                scalar = float(value.mean().cpu())
                row[f"observable_{i}"] = scalar
            results.append(row)
        return results

    def gradient(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[Dict[str, torch.Tensor]]:
        """Return gradients of observables with respect to input parameters using autograd.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Observables that return tensors requiring gradients.
        parameter_sets : Sequence[Sequence[float]]
            Input parameter sets.

        Returns
        -------
        List[Dict[str, torch.Tensor]]
            Gradients for each observable, keyed by observable index.
        """
        self.model.train()  # Enable gradients
        gradients: List[Dict[str, torch.Tensor]] = []
        for params in parameter_sets:
            inputs = _ensure_batch(params).requires_grad_(True).to(self.device)
            outputs = self.model(inputs)
            grads_row: Dict[str, torch.Tensor] = {}
            for i, observable in enumerate(observables):
                value = observable(outputs)
                value.backward(retain_graph=True)
                grads_row[f"observable_{i}"] = inputs.grad.clone().cpu()
                inputs.grad.zero_()
            gradients.append(grads_row)
        self.model.eval()
        return gradients


__all__ = ["FastBaseEstimator"]
