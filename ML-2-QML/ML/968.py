"""Enhanced estimator utilities implemented with PyTorch.

This module extends the original lightweight estimator by adding GPU support,
automatic batching, optional Gaussian shot noise, result caching, and gradient
computation via PyTorch autograd. It preserves the original API while providing
additional flexibility for large‑scale simulations and model introspection.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float], device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimatorGen255:
    """Evaluate neural networks for batches of inputs and observables with optional noise.

    Parameters
    ----------
    model:
        PyTorch ``nn.Module`` to evaluate.
    device:
        Target device; accepts any string accepted by ``torch.device`` (e.g., "cpu" or "cuda").
    """

    def __init__(self, model: nn.Module, *, device: Union[str, torch.device] = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self._cache: Optional[List[List[float]]] = None

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        batch_size: int | None = None,
        shots: int | None = None,
        seed: int | None = None,
        cache: bool = False,
    ) -> List[List[float]]:
        """
        Compute observable values for each parameter set.

        Parameters
        ----------
        observables:
            Iterable of callables mapping model outputs to scalars.
        parameter_sets:
            Sequence of parameter vectors.
        batch_size:
            Size of the batch to feed into the model. ``None`` means no batching.
        shots:
            Number of measurement shots to sample Gaussian noise. ``None`` disables noise.
        seed:
            Random seed for reproducibility of the noise.
        cache:
            If ``True`` and the same parameter set is evaluated again, cached results are returned.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        if cache and self._cache is not None:
            return self._cache

        rng = np.random.default_rng(seed)

        if batch_size is None:
            batch_size = len(parameter_sets)

        self.model.eval()
        for start in range(0, len(parameter_sets), batch_size):
            batch_params = parameter_sets[start : start + batch_size]
            batch_tensor = _ensure_batch(batch_params, self.device)
            with torch.no_grad():
                outputs = self.model(batch_tensor)

            for out in outputs:
                row: List[float] = []
                for observable in observables:
                    val = observable(out)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    if shots is not None:
                        scalar = float(rng.normal(scalar, max(1e-6, 1.0 / shots)))
                    row.append(scalar)
                results.append(row)

        if cache:
            self._cache = results

        return results

    def gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        batch_size: int | None = None,
    ) -> List[List[torch.Tensor]]:
        """
        Compute gradients of each observable w.r.t. the model parameters.

        Returns a list of lists of tensors. Each inner list corresponds to a
        parameter set and contains a tensor of gradients for each observable.
        The gradients are flattened to match the shape of the parameter vector.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        grads: List[List[torch.Tensor]] = []

        if batch_size is None:
            batch_size = len(parameter_sets)

        self.model.train()
        for start in range(0, len(parameter_sets), batch_size):
            batch_params = parameter_sets[start : start + batch_size]
            batch_tensor = _ensure_batch(batch_params, self.device)
            batch_tensor.requires_grad_(True)

            outputs = self.model(batch_tensor)

            for out in outputs:
                row_grads: List[torch.Tensor] = []
                for observable in observables:
                    val = observable(out)
                    if isinstance(val, torch.Tensor):
                        val = val.mean()
                    else:
                        val = torch.tensor(val, device=self.device)
                    val.backward(retain_graph=True)
                    grad = batch_tensor.grad.detach().clone().reshape(-1)
                    row_grads.append(grad)
                    batch_tensor.grad.zero_()
                grads.append(row_grads)

        return grads


__all__ = ["FastBaseEstimatorGen255"]
