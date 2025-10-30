"""Hybrid kernel estimator with classical RBF kernel and FastEstimator utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class KernalAnsatz(nn.Module):
    """Radial‑basis function (RBF) ansatz."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Wrapper around :class:`KernalAnsatz`."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def kernel_matrix(
    a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0
) -> np.ndarray:
    """Compute the Gram matrix between two batches of tensors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridKernelEstimator:
    """
    Classical kernel estimator that supports RBF kernels and fast batched
    evaluation of observables on neural networks.

    Parameters
    ----------
    gamma : float, optional
        RBF kernel width. Defaults to 1.0.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        self.kernel = Kernel(gamma)

    def kernel_matrix(self, X: Sequence[torch.Tensor], Y: Sequence[torch.Tensor]) -> np.ndarray:
        """Return the Gram matrix for two input sequences."""
        return kernel_matrix(X, Y, gamma=self.kernel.ansatz.gamma)

    def evaluate(
        self,
        model: nn.Module,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Evaluate a neural network model on a list of parameter sets and
        observables, optionally adding Gaussian shot noise.

        Parameters
        ----------
        model : nn.Module
            The network to evaluate.
        observables : iterable of callables
            Each callable takes the model output tensor and returns a
            scalar (tensor or float).
        parameter_sets : sequence of parameter sequences
            Each inner sequence contains the parameters for one evaluation.
        shots : int, optional
            Number of shots for noisy estimation. If None, deterministic
            evaluation is performed.
        seed : int, optional
            Random seed for reproducible noise.

        Returns
        -------
        List[List[float]]
            Nested list of observable values for each parameter set.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "HybridKernelEstimator",
]
