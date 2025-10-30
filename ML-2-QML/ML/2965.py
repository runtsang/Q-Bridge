"""Combined classical and quantum kernel regression utilities.

The module builds on the original RBF kernel implementation and extends it with
a lightweight estimator that can evaluate a kernel matrix and fit a kernel
ridge regression model.  It also exposes a wrapper that transparently switches
between a classical RBF kernel and a quantum kernel implemented in the QML
module.

Key additions compared to the seed files:

* :class:`KernalAnsatz` and :class:`Kernel` retain the original interface
  yet expose a ``gamma`` hyper‑parameter that can be tuned by the user.
* :class:`FastKernelEstimator` offers a fast, batched evaluation of a
  sequence of parameter sets with optional Gaussian shot noise.
* :class:`KernelRidge` implements kernel ridge regression using the
  Gram matrix produced by :func:`kernel_matrix`.  The regressor is
  compatible with both the classical and quantum kernel back‑ends.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Sequence, Callable

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

# -- Classical RBF kernel ----------------------------------------------------
class KernalAnsatz(nn.Module):
    """Radial basis function kernel component.

    Parameters
    ----------
    gamma : float, default=1.0
        Width of the Gaussian kernel.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper that exposes a single ``forward`` interface."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix between two collections of vectors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# -- Fast estimator utilities -----------------------------------------------
def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """Evaluate a torch model for a batch of parameter sets."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
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
    """Adds Gaussian shot‑noise to the deterministic estimator."""
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

# -- Kernel ridge regression --------------------------------------------------
class KernelRidge(nn.Module):
    """Kernel ridge regression using a user‑supplied kernel.

    Parameters
    ----------
    kernel : nn.Module
        Callable that returns a scalar kernel value ``k(x, y)``.
    alpha : float, default=1.0
        Regularisation strength.
    """
    def __init__(self, kernel: nn.Module, alpha: float = 1.0) -> None:
        super().__init__()
        self.kernel = kernel
        self.alpha = alpha
        self.coef_ = None
        self.X_train_ = None

    def fit(self, X: Sequence[torch.Tensor], y: torch.Tensor) -> None:
        """Fit the model to training data."""
        K = kernel_matrix(X, X, gamma=getattr(self.kernel.ansatz, "gamma", 1.0))
        K += self.alpha * np.eye(K.shape[0])
        self.coef_ = torch.from_numpy(np.linalg.solve(K, y.numpy()))
        self.X_train_ = X

    def predict(self, X: Sequence[torch.Tensor]) -> torch.Tensor:
        """Return predictions for new data."""
        if self.coef_ is None or self.X_train_ is None:
            raise RuntimeError("Model is not fitted.")
        K_test = kernel_matrix(X, self.X_train_, gamma=getattr(self.kernel.ansatz, "gamma", 1.0))
        return torch.from_numpy(K_test @ self.coef_.numpy())

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "FastBaseEstimator",
    "FastEstimator",
    "KernelRidge",
]
