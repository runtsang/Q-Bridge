import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedSamplerQNN(nn.Module):
    """
    A flexible classical sampler network.
    Parameters
    ----------
    input_dim : int, default 2
        Dimension of input feature space.
    hidden_dims : Sequence[int], default (8, 8)
        Sizes of hidden layers.
    output_dim : int, default 2
        Dimension of output probability distribution.
    dropout : float, default 0.1
        Dropout probability after each hidden layer.
    """
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dims: tuple[int,...] = (8, 8),
                 output_dim: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Draw samples from the categorical distribution."""
        probs = self.forward(x).detach().cpu()
        return torch.multinomial(probs, num_samples=n_samples)

__all__ = ["EnhancedSamplerQNN"]
