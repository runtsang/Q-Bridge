"""Enhanced QCNN model with configurable depth, residual connections, dropout, and feature importance."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class QCNNModel(nn.Module):
    """
    A flexible, depth‑controlled convolution‑inspired network.

    Parameters
    ----------
    input_dim : int
        Size of the input vector.
    hidden_dims : tuple[int,...]
        List of hidden layer sizes. The network will build a block for each element.
    dropout : float | None, optional
        Dropout probability applied after each block. If ``None`` no dropout is used.
    use_residual : bool, optional
        If True, add a residual connection between successive blocks.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: tuple[int,...] = (16, 16, 12, 8, 4, 4),
        dropout: float | None = 0.1,
        use_residual: bool = False,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            block = nn.Sequential(
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.Tanh(),
            )
            if dropout:
                block.append(nn.Dropout(dropout))
            layers.append(block)
            in_dim = h_dim
        self.blocks = nn.ModuleList(layers)
        self.residual = use_residual
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.blocks:
            residual = out
            out = block(out)
            if self.residual:
                out = out + residual  # simple residual addition
        return torch.sigmoid(self.head(out))

    def integrated_gradients(
        self,
        inputs: torch.Tensor,
        target: int = 1,
        steps: int = 50,
    ) -> torch.Tensor:
        """
        Approximate integrated gradients for feature importance.

        Parameters
        ----------
        inputs : torch.Tensor
            Input batch.
        target : int
            Index of the output neuron to attribute.
        steps : int
            Number of integration steps.

        Returns
        -------
        torch.Tensor
            Attribution scores of shape (batch, input_dim).
        """
        baseline = torch.zeros_like(inputs, requires_grad=True)
        scaled_inputs = [
            baseline + (float(i) / steps) * (inputs - baseline) for i in range(1, steps + 1)
        ]
        grads = []
        for inp in scaled_inputs:
            inp.requires_grad_(True)
            out = self.forward(inp)
            out[:, target].backward(retain_graph=True)
            grads.append(inp.grad.clone())
        avg_grad = torch.stack(grads).mean(dim=0)
        return (inputs - baseline) * avg_grad

def QCNN() -> QCNNModel:
    """Factory that returns a default-configured :class:`QCNNModel`."""
    return QCNNModel()

__all__ = ["QCNNModel", "QCNN"]
