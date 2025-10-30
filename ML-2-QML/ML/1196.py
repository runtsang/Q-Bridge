"""Hybrid RBF kernel with trainable parameters and optional bias.

This module extends the original classical RBF kernel by adding a learnable
width parameter (gamma) and an optional bias term.  The kernel is used in
classical machine learning pipelines such as SVMs or Gaussian processes.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class QuantumKernelMethod(nn.Module):
    """Trainable RBF kernel with optional bias and log‑scale.

    Parameters
    ----------
    gamma : float or torch.nn.Parameter, optional
        Width of the RBF kernel. If ``trainable_gamma`` is ``True``, this
        becomes a learnable parameter.
    bias : float or torch.nn.Parameter, optional
        Constant bias added to the kernel.  Can be fixed or learnable.
    log_scale : bool, optional
        If ``True`` the kernel output is ``log1p``‑transformed for numerical
        stability when values become very close to 1.
    trainable_gamma : bool, optional
        Whether ``gamma`` should be a learnable parameter.
    trainable_bias : bool, optional
        Whether ``bias`` should be a learnable parameter.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        bias: float = 0.0,
        log_scale: bool = False,
        trainable_gamma: bool = True,
        trainable_bias: bool = False,
    ) -> None:
        super().__init__()
        self.log_scale = log_scale

        if trainable_gamma:
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        else:
            self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))

        if trainable_bias:
            self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        else:
            self.register_buffer("bias", torch.tensor(bias, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the kernel value for two input vectors.

        Parameters
        ----------
        x, y : torch.Tensor
            Input tensors of shape ``(d,)`` or ``(batch, d)``.  If a batch
            dimension is present, the kernel is evaluated element‑wise.
        """
        # Ensure 2‑D shape for broadcasting
        x = x.unsqueeze(0) if x.dim() == 1 else x
        y = y.unsqueeze(0) if y.dim() == 1 else y

        diff = x - y
        sq_norm = (diff * diff).sum(dim=-1, keepdim=True)
        k = torch.exp(-self.gamma * sq_norm) + self.bias

        if self.log_scale:
            k = torch.log1p(k)

        return k.squeeze(-1)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return the Gram matrix between two collections of samples."""
        a = torch.stack(a).float()
        b = torch.stack(b).float()
        K = torch.zeros((len(a), len(b)), dtype=torch.float32, device=a.device)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                K[i, j] = self.forward(x, y)
        return K.cpu().numpy()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(gamma={self.gamma.item():.4f}, "
            f"bias={self.bias.item():.4f}, log_scale={self.log_scale})"
        )
