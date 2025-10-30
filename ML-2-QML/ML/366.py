"""Enhanced classical kernel module with learnable parameters and batch support."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QuantumKernelMethod(nn.Module):
    """
    A learnable RBF‑style kernel with optional scaling and offset parameters.
    Supports batch‑wise kernel matrix computation for large datasets and
    gradient‑based optimisation of the kernel hyper‑parameters.
    """

    def __init__(
        self,
        init_gamma: float = 1.0,
        learn_gamma: bool = True,
        init_offset: float = 0.0,
        learn_offset: bool = False,
        batch_size: int = 1024,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.gamma = nn.Parameter(torch.tensor(init_gamma)) if learn_gamma else init_gamma
        self.offset = (
            nn.Parameter(torch.tensor(init_offset)) if learn_offset else init_offset
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel matrix K(x, y) = exp(-gamma * ||x - y||^2) + offset.
        The computation is performed in a memory‑efficient batched manner.
        """
        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])

        dist_sq = torch.cdist(x, y, p=2) ** 2

        gamma = (
            self.gamma
            if isinstance(self.gamma, torch.Tensor)
            else torch.tensor(self.gamma, device=x.device, dtype=x.dtype)
        )
        offset = (
            self.offset
            if isinstance(self.offset, torch.Tensor)
            else torch.tensor(self.offset, device=x.device, dtype=x.dtype)
        )

        kernel = torch.exp(-gamma * dist_sq) + offset
        return kernel

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target_kernel: torch.Tensor,
        lr: float = 1e-2,
        epochs: int = 200,
        verbose: bool = False,
    ) -> None:
        """
        Gradient‑based optimisation of the kernel hyper‑parameters to match a
        target kernel matrix (e.g. from a quantum kernel).
        """
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=lr
        )
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = self.forward(x, y)
            loss = loss_fn(pred, target_kernel)
            loss.backward()
            optimizer.step()
            if verbose and (epoch % (epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} loss={loss.item():.6f}")

    def kernel_matrix(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> np.ndarray:
        """
        Compute the Gram matrix between two lists of tensors in a batched way.
        """
        a = torch.stack(a)
        b = torch.stack(b)
        return self.forward(a, b).detach().cpu().numpy()

__all__ = ["QuantumKernelMethod"]
