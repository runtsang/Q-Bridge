from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastEstimator:
    """Evaluate PyTorch neural networks for batches of inputs and observables with optional GPU support and shot noise.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.
    device : torch.device | str | None, optional
        Device on which to run the model. If None, defaults to CUDA if available.
    """

    def __init__(self, model: nn.Module, device: torch.device | str | None = None) -> None:
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Deterministically evaluate observables for each parameter set.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Functions that map model output tensors to scalars.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors.

        Returns
        -------
        List[List[float]]
            A matrix of shape (len(parameter_sets), len(observables)).
        """
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

    def evaluate_and_grad(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        retain_graph: bool = False,
    ) -> tuple[List[List[float]], List[List[torch.Tensor]]]:
        """Evaluate observables and compute gradients w.r.t. model parameters.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
        parameter_sets : Sequence[Sequence[float]]
        retain_graph : bool, optional
            Whether to retain the computational graph for multiple backward passes.

        Returns
        -------
        tuple
            (values, gradients) where values is the same shape as ``evaluate`` and gradients
            is a list of lists of tensors matching the parameter dimensions.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        values: List[List[float]] = []
        grads: List[List[torch.Tensor]] = []

        self.model.train()
        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device)
            inputs.requires_grad = True
            outputs = self.model(inputs)

            row_vals: List[float] = []
            row_grads: List[torch.Tensor] = []
            for observable in observables:
                value = observable(outputs)
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, dtype=outputs.dtype, device=self.device)
                scalar = value.mean()
                row_vals.append(float(scalar.cpu()))
                self.model.zero_grad()
                scalar.backward(retain_graph=retain_graph)
                grad = [p.grad.clone().detach() for p in self.model.parameters() if p.grad is not None]
                row_grads.append(torch.cat([g.view(-1) for g in grad]) if grad else torch.tensor([]))
            values.append(row_vals)
            grads.append(row_grads)
        return values, grads

    def evaluate_noisy(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Add Gaussian shot noise to deterministic results.

        Parameters
        ----------
        shots : int | None
            Number of measurement shots; if None, no noise added.
        seed : int | None
            Random seed for reproducibility.

        Returns
        -------
        List[List[float]]
        """
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastEstimator"]
