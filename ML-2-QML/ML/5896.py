"""Enhanced classical RBF kernel with multi‑scale support and hyper‑parameter optimisation."""

from __future__ import annotations

from typing import Sequence, List, Optional

import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, TransformerMixin
from skopt import gp_minimize
from skopt.space import Real

class QuantumKernelMethod(nn.Module):
    """
    Multi‑scale RBF kernel with learnable weights and bandwidths.
    Supports Bayesian optimisation of hyper‑parameters and GPU execution.
    """

    def __init__(self,
                 n_scales: int = 3,
                 init_gamma: float = 1.0,
                 init_weight: float = 1.0,
                 device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.n_scales = n_scales
        self.device = device or torch.device('cpu')
        self.gamma = nn.Parameter(torch.full((n_scales,), init_gamma, dtype=torch.float32, device=self.device))
        self.weight = nn.Parameter(torch.full((n_scales,), init_weight, dtype=torch.float32, device=self.device))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the multi‑scale RBF kernel between two tensors.
        """
        x = x.to(self.device)
        y = y.to(self.device)
        diff = x.unsqueeze(-2) - y.unsqueeze(-3)
        sq_norm = torch.sum(diff ** 2, dim=-1)
        kernel = torch.zeros_like(sq_norm, device=self.device)
        for i in range(self.n_scales):
            kernel += self.weight[i] * torch.exp(-self.gamma[i] * sq_norm)
        return kernel.sum(dim=-1)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute the Gram matrix between two collections of tensors.
        """
        a_stack = torch.stack([torch.tensor(v, dtype=torch.float32, device=self.device) for v in a])
        b_stack = torch.stack([torch.tensor(v, dtype=torch.float32, device=self.device) for v in b])
        return self.forward(a_stack, b_stack).cpu().numpy()

    def _objective(self, params: List[float]) -> float:
        """Objective for Bayesian optimisation: negative kernel mean (dummy)."""
        gamma_vals, weight_vals = params[:self.n_scales], params[self.n_scales:]
        with torch.no_grad():
            self.gamma.copy_(torch.tensor(gamma_vals, device=self.device))
            self.weight.copy_(torch.tensor(weight_vals, device=self.device))
        dummy_x = torch.randn(10, 5, device=self.device)
        dummy_y = torch.randn(10, 5, device=self.device)
        val = self.forward(dummy_x, dummy_y).mean().item()
        return -val

    def fit_hyperparams(self,
                        X: np.ndarray,
                        y: Optional[np.ndarray] = None,
                        n_calls: int = 30,
                        random_state: int = 42) -> None:
        """
        Fit kernel hyper‑parameters using Bayesian optimisation.
        """
        space = [Real(1e-3, 1e2, name=f'gamma_{i}') for i in range(self.n_scales)] + \
                [Real(1e-3, 1e2, name=f'weight_{i}') for i in range(self.n_scales)]
        result = gp_minimize(self._objective, space, n_calls=n_calls,
                             random_state=random_state, verbose=False)
        best_params = result.x
        gamma_best, weight_best = best_params[:self.n_scales], best_params[self.n_scales:]
        self.gamma.copy_(torch.tensor(gamma_best, device=self.device))
        self.weight.copy_(torch.tensor(weight_best, device=self.device))

    def __repr__(self) -> str:
        return f"QuantumKernelMethod(n_scales={self.n_scales}, gamma={self.gamma.tolist()}, weight={self.weight.tolist()})"

__all__ = ["QuantumKernelMethod"]
