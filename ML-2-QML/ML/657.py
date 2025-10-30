import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """Extended classical sampler network with optional dropout and batch‑norm layers."""
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 32,
                 output_dim: int = 2,
                 dropout: float = 0.1,
                 use_batchnorm: bool = False):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a probability distribution over the output classes."""
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Sample discrete outputs according to the softmax distribution."""
        probs = self.forward(x)
        dist = torch.distributions.Categorical(probs)
        return dist.sample((num_samples,))

    def kl_divergence(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence KL(p || q)."""
        eps = 1e-12
        p = p + eps
        q = q + eps
        return torch.sum(p * torch.log(p / q), dim=-1)

__all__ = ["SamplerQNN"]
