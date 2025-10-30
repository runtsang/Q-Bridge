"""Enhanced classical RBF kernel utilities with batched support and hyper‑parameter tuning."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence, Iterable, Optional, Tuple


class RBFKernel(nn.Module):
    """
    A batched, trainable RBF kernel.

    Parameters
    ----------
    gamma : float or nn.Parameter
        Width of the Gaussian. If `trainable` is True, this is wrapped in an
        ``nn.Parameter`` and optimized by gradient descent.
    trainable : bool, default=False
        Whether ``gamma`` should be treated as a learnable parameter.
    feature_map : Optional[Callable[[torch.Tensor], torch.Tensor]]
        Optional transformation applied to the input before computing the kernel.
        Useful for dimensionality expansion or nonlinear embeddings.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        trainable: bool = False,
        feature_map: Optional[callable] = None,
    ) -> None:
        super().__init__()
        if trainable:
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        else:
            self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))
        self.feature_map = feature_map

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel between two batches of vectors.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(n_samples_x, n_features)``.
        y : torch.Tensor
            Shape ``(n_samples_y, n_features)``.

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape ``(n_samples_x, n_samples_y)``.
        """
        if self.feature_map is not None:
            x = self.feature_map(x)
            y = self.feature_map(y)
        # Expand dimensions for broadcasting
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # shape (n_x, n_y, d)
        sq_norm = torch.sum(diff * diff, dim=-1)  # shape (n_x, n_y)
        return torch.exp(-self.gamma * sq_norm)


def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    gamma: float = 1.0,
    trainable: bool = False,
    feature_map: Optional[callable] = None,
) -> np.ndarray:
    """
    Compute the Gram matrix between two collections of tensors using the RBF kernel.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Each element should be a 1‑D tensor representing a data point.
    gamma : float, default=1.0
        Kernel width.
    trainable : bool, default=False
        Whether ``gamma`` should be trainable.
    feature_map : Optional[Callable], default=None
        Optional feature transformation.

    Returns
    -------
    np.ndarray
        Kernel matrix as a NumPy array.
    """
    kernel = RBFKernel(gamma=gamma, trainable=trainable, feature_map=feature_map)
    a_batch = torch.stack(a)
    b_batch = torch.stack(b)
    return kernel(a_batch, b_batch).cpu().numpy()


def cross_validate_gamma(
    data: Sequence[torch.Tensor],
    labels: Sequence[int],
    gamma_values: Iterable[float],
    cv: int = 5,
) -> float:
    """
    Simple grid search for the best gamma using leave‑one‑out cross‑validation
    with a support‑vector‑machine objective.

    Parameters
    ----------
    data : Sequence[torch.Tensor]
        Data points.
    labels : Sequence[int]
        Corresponding class labels.
    gamma_values : Iterable[float]
        Candidate gamma values.
    cv : int, default=5
        Number of folds.

    Returns
    -------
    float
        The gamma that maximizes the cross‑validated accuracy.
    """
    import sklearn.svm
    from sklearn.model_selection import cross_val_score

    X = torch.stack(data).cpu().numpy()
    y = np.array(labels)

    best_gamma = gamma_values[0]
    best_score = -np.inf
    for gamma in gamma_values:
        clf = sklearn.svm.SVC(kernel="rbf", gamma=gamma)
        score = cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean()
        if score > best_score:
            best_score = score
            best_gamma = gamma
    return best_gamma


__all__ = ["RBFKernel", "kernel_matrix", "cross_validate_gamma"]
