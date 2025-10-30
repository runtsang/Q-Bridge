"""Enhanced fully‑connected layer with batched inference and hybrid training support."""

from __future__ import annotations

import torch
from torch import nn
from torch.autograd import grad
from typing import Sequence, Tuple

class FCL(nn.Module):
    """
    A fully‑connected layer that can process a batch of inputs and
    optionally compute gradients for both a classical linear map
    and a quantum‑parameterized circuit.
    """

    def __init__(self, n_features: int = 1, *, heads: int = 1, device: str = "cpu"):
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=False)
        self.qparams = nn.Parameter(torch.randn(heads, n_features, device=device))

    def forward(self, x: torch.Tensor, thetas: Sequence[torch.Tensor]) -> torch.Tensor:
        classical_out = self.linear(x)
        quantum_out = torch.stack(
            [torch.tanh((x @ theta).unsqueeze(-1)) for theta in thetas],
            dim=-1,
        ).mean(dim=-1)
        return classical_out + quantum_out

    def compute_gradients(self, x: torch.Tensor, thetas: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor,...]]:
        output = self.forward(x, thetas)
        loss = output.mean()
        grads = grad(loss, [self.linear.weight, *thetas], retain_graph=True)
        return grads[0], grads[1:]

__all__ = ["FCL"]
