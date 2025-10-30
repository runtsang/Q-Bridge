"""Hybrid kernel and estimator combining classical RBF and quantum kernel with shot‑noise support.

This module defines :class:`HybridKernelMethod` that implements:
* classical RBF kernel computation,
* a quantum kernel wrapper that can be injected,
* a lightweight estimator for neural networks with optional Gaussian shot noise.

The interface mirrors the original seed modules but unifies the two back‑ends.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Callable, Iterable, List, Sequence

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridKernelMethod(nn.Module):
    """Hybrid kernel and estimator with classical RBF kernel and optional quantum kernel.

    Parameters
    ----------
    gamma : float, optional
        Width parameter for the RBF kernel.
    model : nn.Module | None, optional
        Neural network model used by the estimator.
    quantum_kernel : Callable[[Sequence[torch.Tensor], Sequence[torch.Tensor]], np.ndarray] | None, optional
        Callable that returns a quantum kernel matrix.  When provided, the
        :meth:`kernel_matrix` method returns a weighted combination of the
        classical RBF kernel and the supplied quantum kernel.
    alpha : float, optional
        Weight for the classical kernel in the hybrid combination.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        model: nn.Module | None = None,
        quantum_kernel: Callable[[Sequence[torch.Tensor], Sequence[torch.Tensor]], np.ndarray] | None = None,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.model = model
        self.quantum_kernel = quantum_kernel
        self.alpha = alpha

    # Classical RBF kernel
    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    # Kernel matrix combining classical and optional quantum parts
    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        classical = np.array([[self.rbf_kernel(x, y).item() for y in b] for x in a])
        if self.quantum_kernel is None:
            return classical
        quantum = np.asarray(self.quantum_kernel(a, b))
        return self.alpha * classical + (1.0 - self.alpha) * quantum

    # Estimator with optional shot noise
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate observables on the model for a set of parameters.

        If ``shots`` is provided, Gaussian shot noise is injected.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        if self.model is None:
            raise ValueError("Model must be provided for evaluation.")
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
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["HybridKernelMethod"]
