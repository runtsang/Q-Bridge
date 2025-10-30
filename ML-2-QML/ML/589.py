"""Enhanced classical estimator with flexible architecture and diagnostics."""

from __future__ import annotations

import torch
from torch import nn
from typing import Sequence, Callable, Optional

class EstimatorQNNGen236(nn.Module):
    """
    A flexible feed‑forward regressor.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_layers : Sequence[int]
        Sizes of hidden layers. Empty sequence yields a linear model.
    activation : Callable[[torch.Tensor], torch.Tensor]
        Non‑linearity applied after each hidden layer.
    dropout : float | None
        Dropout probability applied after each hidden layer.
    bias : bool
        Whether each linear layer uses a bias term.
    """
    def __init__(self,
                 input_dim: int = 2,
                 hidden_layers: Sequence[int] = (8, 4),
                 activation: Callable[[torch.Tensor], torch.Tensor] = nn.Tanh(),
                 dropout: Optional[float] = None,
                 bias: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h, bias=bias))
            layers.append(activation)
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the scalar prediction."""
        return self.net(x)

    def hidden_forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return all intermediate activations for inspection."""
        activations: list[torch.Tensor] = []
        h = x
        for layer in self.net:
            h = layer(h)
            if isinstance(layer, nn.Linear):
                activations.append(h)
        return activations

    def get_weights(self) -> dict[str, torch.Tensor]:
        """Return a dict of named weight tensors."""
        return {name: param for name, param in self.named_parameters() if param.requires_grad}

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper around forward."""
        return self.forward(x)

def EstimatorQNN() -> EstimatorQNNGen236:
    """Factory returning a default instance."""
    return EstimatorQNNGen236()

__all__ = ["EstimatorQNNGen236", "EstimatorQNN"]
