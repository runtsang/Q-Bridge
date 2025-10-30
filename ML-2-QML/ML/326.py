"""Enhanced classical RBF kernel with per‑feature widths and optional preprocessing.

This module extends the original implementation by:
* Allowing a separate width (gamma) per feature, stored as a learnable
  torch.nn.Parameter.
* Providing an optional preprocessing hook (e.g. z‑score normalisation)
  that is applied before the kernel computation.
* Exposing a `trainable` flag that can be used by downstream optimisation
  loops (e.g. kernel‑SVM training).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class KernalAnsatz(nn.Module):
    """
    Flexible RBF kernel that can learn a separate width for each feature.

    Parameters
    ----------
    gamma : float | Sequence[float] | torch.Tensor
        Initial width(s). If a scalar is supplied the same width is used
        for all features. If a sequence or tensor is supplied its length
        determines the number of features. The value is wrapped in a
        learnable :class:`torch.nn.Parameter` and passed through a
        ``softplus`` to guarantee positivity during optimisation.
    preprocess : Callable[[torch.Tensor], torch.Tensor] | None
        Optional callable that is applied to both input tensors before the
        kernel evaluation (e.g. feature scaling).
    """

    def __init__(
        self,
        gamma: float | Sequence[float] | torch.Tensor = 1.0,
        *,
        preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        if isinstance(gamma, (float, int)):
            gamma = torch.tensor([float(gamma)], dtype=torch.float32)
        elif isinstance(gamma, Sequence):
            gamma = torch.tensor(gamma, dtype=torch.float32)
        self.gamma = nn.Parameter(gamma)
        self.preprocess = preprocess

    @property
    def positive_gamma(self) -> torch.Tensor:
        """Return a positive version of gamma via softplus."""
        return F.softplus(self.gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel ``exp(-∑ γ_i (x_i - y_i)^2)``.

        Parameters
        ----------
        x, y : torch.Tensor
            Input tensors of shape ``(N, D)`` where ``D`` is the number of
            features.  If ``x`` or ``y`` is 1‑D it is treated as a single
            sample.
        """
        if self.preprocess is not None:
            x = self.preprocess(x)
            y = self.preprocess(y)

        # Ensure tensors are 2‑D
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if y.ndim == 1:
            y = y.unsqueeze(0)

        diff = x.unsqueeze(1) - y.unsqueeze(0)  # shape (Nx, Ny, D)
        sq_diff = diff.pow(2)  # element‑wise squared difference
        weighted = sq_diff * self.positive_gamma  # broadcasting over D
        sum_weighted = weighted.sum(dim=-1)  # shape (Nx, Ny)
        return torch.exp(-sum_weighted)


class Kernel(nn.Module):
    """
    Wrapper that exposes the :class:`KernalAnsatz` and a ``trainable`` flag.

    The flag can be used by downstream code to decide whether the kernel
    parameters should be optimised together with a model (e.g. SVM).
    """

    def __init__(
        self,
        gamma: float | Sequence[float] | torch.Tensor = 1.0,
        *,
        preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma, preprocess=preprocess)
        self.trainable = trainable

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y)


def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    gamma: float | Sequence[float] | torch.Tensor = 1.0,
    *,
    preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> np.ndarray:
    """
    Compute a Gram matrix between two lists of tensors.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of feature vectors.
    gamma : float | Sequence[float] | torch.Tensor
        Width(s) for the RBF kernel.
    preprocess : Callable, optional
        Optional preprocessing applied to each vector before the kernel
        evaluation.
    """
    kernel = Kernel(gamma, preprocess=preprocess)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
