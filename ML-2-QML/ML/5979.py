import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """
    Classical sampler network with a residual connection, dropout,
    and a learnable bias term. The network outputs a probability
    distribution over two classes via softmax.

    The architecture is a lightweight extension of the original
    seed: a linear → tanh → linear block is wrapped in a residual
    path and augmented with dropout and bias to increase capacity.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 4, dropout: float = 0.2):
        super().__init__()
        # Main feed‑forward path
        self.main = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )
        # Residual connection
        self.residual = nn.Linear(input_dim, input_dim)
        # Learnable bias term
        self.bias = nn.Parameter(torch.zeros(input_dim))
        # Dropout for regularisation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing a probability distribution over two outputs.
        """
        main_out = self.main(x)
        res_out = self.residual(x)
        out = main_out + res_out + self.bias
        out = self.dropout(out)
        return F.softmax(out, dim=-1)

__all__ = ["SamplerQNN"]
