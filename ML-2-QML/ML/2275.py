"""Hybrid classical kernel module with optional quantum backend.

The :class:`Kernel` class is a PyTorch ``nn.Module`` that implements a
radial‑basis‑function (RBF) kernel and can optionally delegate the
kernel evaluation to a user supplied quantum kernel.  The module also
exposes a lightweight estimator API that mirrors the
``FastBaseEstimator`` pattern from the original project, but adds
support for batched parameter sets and configurable shot noise.

The public API is intentionally minimal: ``Kernel``, ``KernelAnsatz``
(and its legacy alias ``KernalAnsatz``), ``kernel_matrix``,
``FastBaseEstimator`` and ``FastEstimator``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn


ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a list of scalars into a 2‑D ``torch.Tensor`` of shape
    ``(1, N)``.  The helper is used by the estimator classes to
    support single‑parameter inputs.
    """
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class KernelAnsatz(nn.Module):
    """Pure‑Python RBF ansatz used by :class:`Kernel`.

    The class is kept separate to expose the kernel computation as a
    reusable sub‑module.  It accepts a ``gamma`` hyper‑parameter
    controlling the width of the RBF.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


# Backward‑compatibility alias used by the original seed
KernalAnsatz = KernelAnsatz


class Kernel(nn.Module):
    """Hybrid kernel that defaults to an RBF kernel but can wrap a quantum kernel.

    The ``quantum_kernel`` argument must be a callable that accepts
    two ``torch.Tensor`` arguments and returns a scalar ``torch.Tensor``.
    When provided, the kernel delegates to ``quantum_kernel``; otherwise
    it uses :class:`KernelAnsatz` to compute the classical RBF.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        *,
        quantum_kernel: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.ansatz = KernelAnsatz(gamma)
        self.quantum_kernel = quantum_kernel

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value for ``x`` and ``y``."""
        x = x.view(1, -1)
        y = y.view(1, -1)
        if self.quantum_kernel is None:
            return self.ansatz(x, y).squeeze()
        return self.quantum_kernel(x, y).squeeze()

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """Compute the Gram matrix for two collections of vectors."""
        return np.array([[self(x, y).item() for y in b] for x in a])


def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    gamma: float = 1.0,
) -> np.ndarray:
    """Convenience wrapper that constructs a plain RBF kernel."""
    kernel = Kernel(gamma)
    return kernel.kernel_matrix(a, b)


class FastBaseEstimator:
    """Evaluate a model for a batch of parameter sets and observables.

    The estimator is intentionally lightweight: it runs the model
    in evaluation mode and collects the requested observables.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a 2‑D list of observable values.

        Parameters
        ----------
        observables:
            Callables that map the model output to a scalar.
        parameter_sets:
            A sequence of parameter vectors to evaluate.
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
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Add Gaussian shot noise to the deterministic estimates.

    The ``shots`` argument controls the standard deviation of the noise
    added to each observable value.
    """

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = [
    "Kernel",
    "KernelAnsatz",
    "KernalAnsatz",
    "kernel_matrix",
    "FastBaseEstimator",
    "FastEstimator",
]
