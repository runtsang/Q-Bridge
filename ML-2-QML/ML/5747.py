from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import torch
import numpy as np
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of parameter values into a 2D tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """
    Lightweight, GPU‑accelerable estimator for PyTorch models.

    Features
    --------
    * Batch evaluation on arbitrary device.
    * Optional dropout during inference for aleatoric uncertainty.
    * Vectorised observables with user‑supplied callables.
    * Gradient extraction via torch.autograd for sensitivity analysis.
    """

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        dropout: bool = False,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Evaluate the model for each set of parameters and return a list of observable values.

        Parameters
        ----------
        observables: iterable of callables
            Each callable accepts the model output and returns a scalar or tensor.
        parameter_sets: sequence of parameter sequences
            Each inner sequence contains the values for the model's parameters.
        dropout: bool, optional
            If True, the model is evaluated with dropout layers active.
        seed: int, optional
            Seed for reproducible dropout behaviour.

        Returns
        -------
        List[List[float]]
            A list of rows, one per parameter set, each row containing the observables.
        """
        self.model.eval()
        if dropout:
            torch.manual_seed(seed)
            self.model.train()  # Activates dropout layers

        results: List[List[float]] = []

        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)

        return results

    def evaluate_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[torch.Tensor]]:
        """
        Compute gradients of each observable with respect to the model parameters.

        Returns
        -------
        List[List[torch.Tensor]]
            For each parameter set, a list of gradient tensors matching the shape of
            the corresponding observable. Gradients are evaluated on the same device
            as the model.
        """
        grads: List[List[torch.Tensor]] = []

        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device)
            outputs = self.model(inputs)

            for obs in observables:
                self.model.zero_grad()
                val = obs(outputs)
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val, device=self.device, requires_grad=True)
                val.mean().backward(retain_graph=True)
                # Flatten all parameter gradients into a single vector
                grad_vector = torch.cat([p.grad.flatten() for p in self.model.parameters()]).clone()
                grads.append(grad_vector)

        return grads


__all__ = ["FastBaseEstimator"]
