"""Enhanced lightweight estimator using PyTorch with batched inference, GPU support, and gradient utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D tensor from a 1‑D sequence of floats."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """
    Evaluate a PyTorch neural network for multiple parameter sets.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.
    device : str | torch.device | None, optional
        Target device; defaults to ``"cpu"``.  Pass ``"cuda"`` to enable GPU.

    Notes
    -----
    * The estimator operates in ``eval`` mode and disables gradients.
    * ``evaluate`` returns a 2‑D ``torch.Tensor`` of shape
      ``(n_parameter_sets, n_observables)``.
    * ``compute_gradients`` performs a forward‑backward pass for each
      parameter set and returns gradients for all model parameters.
    """

    def __init__(self, model: nn.Module, device: str | torch.device | None = None) -> None:
        self.model = model
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)
        self.model.eval()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> torch.Tensor:
        """Return expectation values for all observables and parameter sets."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

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

        return torch.tensor(results, dtype=torch.float32, device=self.device)

    def compute_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[torch.Tensor]]:
        """
        Compute gradients of the summed observable loss w.r.t. model parameters
        for each parameter set.

        Returns a list of gradients, one per parameter set.  Each gradient
        element is a list of tensors corresponding to ``model.parameters()``.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        grads: List[List[torch.Tensor]] = []

        for params in parameter_sets:
            self.model.train()
            self.model.zero_grad()

            inputs = _ensure_batch(params).to(self.device)
            outputs = self.model(inputs)

            loss = 0.0
            for observable in observables:
                value = observable(outputs)
                if isinstance(value, torch.Tensor):
                    loss += value.mean()
                else:
                    loss += torch.tensor(value, device=self.device)

            loss.backward()

            grads.append([p.grad.clone().detach() for p in self.model.parameters()])

        self.model.eval()
        return grads

    def evaluate_with_noise(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *, shots: int | None = None, seed: int | None = None,
    ) -> torch.Tensor:
        """
        Evaluate with optional Gaussian shot noise.

        Parameters
        ----------
        shots : int | None
            Number of shots; if ``None`` no noise is added.
        seed : int | None
            RNG seed for reproducibility.
        """
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw

        rng = np.random.default_rng(seed)
        noise_std = max(1e-6, 1 / shots)
        noisy = raw.cpu().numpy() + rng.normal(0, noise_std, raw.shape)
        return torch.from_numpy(noisy).to(self.device)


__all__ = ["FastBaseEstimator"]
