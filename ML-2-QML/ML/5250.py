"""Hybrid classical kernel regression implementation.

This module builds on the classical RBF kernel from `QuantumKernelMethod.py`
and the regression network from `QuantumRegression.py`.  The
`HybridKernelRegression` class exposes a learnable RBF kernel together
with a lightweight feed‑forward head.  It also provides a kernel‑ridge
prediction routine and a fast estimator that can inject Gaussian shot
noise to emulate a noisy quantum device.

The public API mirrors the original seed modules while adding the
following enhancements:

*   `gamma` is now a learnable parameter, regularised via `softplus`
    to keep it positive.
*   The kernel matrix is computed in a fully vectorised way using
    broadcasting, which is faster than the nested list comprehension
    used in the seed.
*   A simple `predict` method implements kernel ridge regression
    without external dependencies.
*   `FastEstimator` accepts an arbitrary number of observable
    callables and supports an optional shot noise model.
"""

import numpy as np
import torch
from torch import nn
from torch.nn.functional import softplus
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastEstimator:
    """Deterministic estimator with optional Gaussian shot noise.

    Parameters
    ----------
    model : nn.Module
        A PyTorch model that maps a batch of inputs to a batch of outputs.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(self,
                 observables: Iterable[ScalarObservable],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        """Evaluate observables for many parameter sets.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Callables that map a model output to a scalar.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors.

        Returns
        -------
        List[List[float]]
            A nested list where each sub‑list contains the observable values
            for a single parameter set.
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
                    scalar = float(value.mean().cpu()) if isinstance(value, torch.Tensor) else float(value)
                    row.append(scalar)
                results.append(row)
        return results

class FastEstimatorWithNoise(FastEstimator):
    """Adds Gaussian shot noise to the deterministic estimator."""

    def evaluate(self,
                 observables: Iterable[ScalarObservable],
                 parameter_sets: Sequence[Sequence[float]],
                 *,
                 shots: int | None = None,
                 seed: int | None = None) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

class HybridKernelRegression(nn.Module):
    """Classical kernel regression with a learnable RBF kernel.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input data.
    hidden_dim : int, optional
        Size of the hidden layer in the regression head.
    gamma_init : float, optional
        Initial value for the RBF width parameter.
    """

    def __init__(self,
                 num_features: int,
                 hidden_dim: int = 32,
                 gamma_init: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))
        self.reg_head = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    # -----------------------------------------------------------------------
    # Kernel utilities

    def rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute a single RBF kernel value.

        The gamma parameter is passed through ``softplus`` to keep it
        strictly positive.
        """
        diff = x - y
        return torch.exp(-softplus(self.gamma) * diff.pow(2).sum(-1, keepdim=True))

    def kernel_matrix(self,
                      a: torch.Tensor,
                      b: torch.Tensor) -> torch.Tensor:
        """Vectorised computation of the Gram matrix between ``a`` and ``b``."""
        a_exp = a.unsqueeze(1).expand(a.shape[0], b.shape[0], a.shape[1])
        b_exp = b.unsqueeze(0).expand(a.shape[0], b.shape[0], b.shape[1])
        return self.rbf(a_exp, b_exp).squeeze(-1)

    # -----------------------------------------------------------------------
    # Prediction utilities

    def predict(self,
                X_train: torch.Tensor,
                y_train: torch.Tensor,
                X_test: torch.Tensor,
                reg_lambda: float = 1e-5) -> torch.Tensor:
        """Kernel ridge regression prediction.

        Parameters
        ----------
        X_train : torch.Tensor
            Training data of shape ``(n, d)``.
        y_train : torch.Tensor
            Training targets of shape ``(n,)``.
        X_test : torch.Tensor
            Test data of shape ``(m, d)``.
        reg_lambda : float, optional
            Ridge regularisation coefficient.

        Returns
        -------
        torch.Tensor
            Predictions of shape ``(m,)``.
        """
        K = self.kernel_matrix(X_train, X_train)
        K += reg_lambda * torch.eye(K.shape[0], device=K.device)
        alpha = torch.linalg.solve(K, y_train)
        K_test = self.kernel_matrix(X_test, X_train)
        return (K_test @ alpha).squeeze(-1)

# ---------------------------------------------------------------------------

# Utility dataset and generation functions (from QuantumRegression.py)

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for regression.

    The targets are a noisy trigonometric function of the sum of the
    input features, mimicking a superposition state.

    Parameters
    ----------
    num_features : int
        Number of input dimensions.
    samples : int
        Number of data points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(features, labels)`` where ``features`` has shape
        ``(samples, num_features)`` and ``labels`` has shape ``(samples,)``.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Simple PyTorch dataset wrapping the synthetic regression data."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

__all__ = [
    "HybridKernelRegression",
    "generate_superposition_data",
    "RegressionDataset",
    "FastEstimator",
    "FastEstimatorWithNoise",
]
