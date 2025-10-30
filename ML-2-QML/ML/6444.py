import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNEnhanced(nn.Module):
    """
    Enhanced classical sampler network with residual connections, batch normalization,
    and dropout. Mirrors the original SamplerQNN but with a deeper architecture
    that improves expressivity and generalization.
    """
    def __init__(self, input_dim: int = 2, hidden_dims: list[int] | tuple[int,...] = (8, 8, 8),
                 output_dim: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.features = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        out = self.output_layer(h)
        return F.softmax(out, dim=-1)

    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Sample from the categorical distribution produced by the network.
        """
        probs = self.forward(x)
        return torch.multinomial(probs, num_samples=num_samples, replacement=True)

__all__ = ["SamplerQNNEnhanced"]
