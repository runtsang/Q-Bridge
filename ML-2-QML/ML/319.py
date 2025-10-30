"""Classical sampler network with deeper architecture and regularisation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNN(nn.Module):
    """
    A multi‑layer perceptron that maps variable‑dimensional inputs to a
    probability distribution over the same number of output classes.
    Regularisation (batch‑norm & dropout) is added to improve generalisation.
    """

    def __init__(self, input_dim: int = 2, hidden_dims: tuple[int,...] = (32, 16), dropout: float = 0.2) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., input_dim)

        Returns
        -------
        torch.Tensor
            Softmax probability distribution of shape (..., input_dim)
        """
        logits = self.net(x)
        return F.softmax(logits, dim=-1)


__all__ = ["SamplerQNN"]
