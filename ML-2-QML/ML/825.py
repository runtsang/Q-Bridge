import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNModel(nn.Module):
    """Classical feed‑forward sampler network.

    Parameters
    ----------
    input_dim : int
        Dimension of input feature vector.
    hidden_dim : int, optional
        Size of hidden layer. Defaults to 32.
    output_dim : int, optional
        Size of output probability vector. Defaults to 2.
    dropout : float, optional
        Dropout probability. Defaults to 0.0.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 32,
                 output_dim: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.extend([
            nn.Linear(hidden_dim, output_dim),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability distribution via softmax."""
        return F.softmax(self.net(x), dim=-1)

    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Draw samples from the categorical distribution defined by the network."""
        probs = self.forward(x)
        return torch.multinomial(probs, n_samples, replacement=True)

__all__ = ["SamplerQNNModel"]
