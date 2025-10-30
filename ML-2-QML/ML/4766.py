"""Hybrid estimator combining classical neural network evaluation and kernel utilities.

This module defines :class:`HybridBaseEstimator` that accepts a PyTorch ``nn.Module``,
provides deterministic and noisy evaluation of observables, and exposes a
classical RBF kernel interface.  The design mirrors the quantum‑side
implementation, enabling side‑by‑side experiments.
"""

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import List, Callable, Optional

# --------------------------------------------------------------------------- #
# 1. Classic kernel utilities
# --------------------------------------------------------------------------- #

class KernalAnsatz(nn.Module):
    """Radial basis function kernel implementation."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper that normalises input shapes for a single‑sample kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix between two collections of tensors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# 2. Default classical model (Quantum‑NAT style)
# --------------------------------------------------------------------------- #

class QFCModel(nn.Module):
    """Classical CNN + FC projection used as a default model."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

# --------------------------------------------------------------------------- #
# 3. Hybrid estimator
# --------------------------------------------------------------------------- #

class HybridBaseEstimator:
    """Evaluate a PyTorch model over parameter sets and observables.

    Parameters
    ----------
    model : nn.Module
        Any differentiable PyTorch model.  By default a lightweight CNN
        (:class:`QFCModel`) is used.
    kernel : Callable | None
        Optional callable that accepts two tensors and returns a scalar.
        If omitted, the RBF kernel defined above is used.
    """
    def __init__(self, model: nn.Module | None = None,
                 kernel: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None) -> None:
        self.model = model or QFCModel()
        self.kernel = kernel or (lambda x, y: kernel_matrix([x], [y], gamma=1.0)[0, 0])

    # --------------------------------------------------------------------- #
    # Evaluation helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def evaluate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: Sequence[Sequence[float]],
                 *,
                 shots: int | None = None,
                 seed: int | None = None) -> List[List[float]]:
        """Compute deterministic or noisy expectations.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output and returns a scalar.
        parameter_sets : sequence of parameter vectors
            Each vector is fed to ``self.model`` as a single‑sample batch.
        shots, seed : optional
            If ``shots`` is given, Gaussian noise with variance ``1/shots``
            is added to each expectation.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = self._ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    val = observable(outputs)
                    scalar = float(val.mean().cpu()) if isinstance(val, torch.Tensor) else float(val)
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

    def evaluate_kernels(self,
                         x: Sequence[torch.Tensor],
                         y: Sequence[torch.Tensor]) -> np.ndarray:
        """Return Gram matrix using the configured kernel."""
        return np.array([[self.kernel(a, b).item() for b in y] for a in x])

__all__ = ["HybridBaseEstimator", "Kernel", "kernel_matrix", "QFCModel"]
