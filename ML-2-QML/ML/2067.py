"""Extended classical sampler network mirroring the original SamplerQNN.

Features:
- Configurable input dimension and hidden layers.
- Optional dropout for regularisation.
- Utility method `predict_proba` for inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNN(nn.Module):
    """A configurable feed‑forward sampler network.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input tensor.
    hidden_dims : Sequence[int], optional
        List of hidden layer sizes. Defaults to ``[4]``.
    dropout : float or None, optional
        Dropout probability applied after each hidden layer. If ``None``, no dropout.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list[int] | tuple[int,...] | None = None,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [4]
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class‑probability vector (softmax)."""
        return F.softmax(self.net(inputs), dim=-1)

    def predict_proba(self, inputs: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper returning probabilities."""
        return self.forward(inputs)


__all__ = ["SamplerQNN"]
