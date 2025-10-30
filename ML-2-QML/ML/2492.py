"""Hybrid estimator combining a PyTorch neural network with optional shot noise and kernel methods.

The class exposes a fast evaluation interface for batches of parameters and
scalar observables, and implements a radial‑basis function kernel
compatible with the quantum interface.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn


ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a list of floats into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastHybridEstimator:
    """
    Evaluate a PyTorch model for many parameter sets and observables.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.  It must accept a 2‑D tensor of shape
        ``(batch, features)`` and return a 2‑D tensor of shape
        ``(batch, outputs)``.
    noise_shots : int | None, optional
        If provided, the estimator adds Gaussian shot noise with variance
        ``1 / shots`` to each observable.  This mimics the behaviour of a
        quantum measurement device.
    noise_seed : int | None, optional
        Seed for the pseudo‑random generator used to generate shot noise.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        noise_shots: int | None = None,
        noise_seed: int | None = None,
    ) -> None:
        self.model = model
        self.noise_shots = noise_shots
        self.noise_seed = noise_seed
        if noise_shots is not None:
            self._rng = np.random.default_rng(noise_seed)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Compute the observables for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output and returns a scalar
            (tensor or float).  If empty, the mean of the last dimension is
            used.
        parameter_sets : sequence of sequences
            Each inner sequence contains the parameters for one evaluation.

        Returns
        -------
        List[List[float]]
            A list of rows, one per parameter set, each containing the
            values of all observables.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)

                if self.noise_shots is not None:
                    row = [
                        float(self._rng.normal(mean, max(1e-6, 1 / self.noise_shots)))
                        for mean in row
                    ]

                results.append(row)

        return results

    # ------------------------------------------------------------------
    # Kernel helpers – classical RBF, quantum‑compatible interface
    # ------------------------------------------------------------------
    class _RBFKernel(nn.Module):
        """Internal RBF kernel implementation."""

        def __init__(self, gamma: float = 1.0) -> None:
            super().__init__()
            self.gamma = gamma

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            diff = x - y
            return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
        gamma: float = 1.0,
    ) -> np.ndarray:
        """
        Compute the Gram matrix using a radial‑basis function.

        Parameters
        ----------
        a, b : sequences of tensors
            Each tensor must be 1‑D and represent a data point.
        gamma : float, optional
            Width parameter of the RBF kernel.

        Returns
        -------
        np.ndarray
            The kernel matrix of shape ``(len(a), len(b))``.
        """
        kernel = self._RBFKernel(gamma)
        return np.array(
            [[kernel(x, y).item() for y in b] for x in a]
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r}, noise_shots={self.noise_shots})"


__all__ = ["FastHybridEstimator"]
