import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNGen124(nn.Module):
    """
    Extended sampler network with optional dropout, batch norm, and
    sampling utilities. Mirrors the original 2→4→2 architecture but
    supports variable input dimensionality via a simple linear
    projection. Provides methods to return logits, probabilities,
    and to sample from the categorical distribution.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 4, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits before softmax."""
        return self.net(x)

    def sample(self, x: torch.Tensor, num_samples: int = 1, device: torch.device | None = None) -> torch.Tensor:
        """
        Draw samples from the categorical distribution defined by the
        network's output probabilities.
        """
        probs = self.forward(x)
        probs = probs.to(device) if device else probs
        return torch.multinomial(probs, num_samples, replacement=True)

    def to_torchscript(self):
        """Return a TorchScript version of the model."""
        return torch.jit.script(self)

__all__ = ["SamplerQNNGen124"]
