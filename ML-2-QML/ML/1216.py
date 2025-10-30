import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedSamplerQNN(nn.Module):
    """
    A richer classical sampler network.
    - Configurable input dimension.
    - Three hidden layers with batch normalization, ReLU, and dropout.
    - Outputs a probability distribution via softmax.
    - Provides a sample() method for drawing from the distribution.
    """
    def __init__(self, input_dim: int = 2, hidden_dims: list[int] | None = None,
                 output_dim: int = 2, dropout: float = 0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [8, 8, 8]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h),
                           nn.BatchNorm1d(h),
                           nn.ReLU(),
                           nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def sample(self, x: torch.Tensor, num_samples: int = 1,
               device: str | None = None) -> torch.Tensor:
        probs = self.forward(x)
        probs = probs if device is None else probs.to(device)
        return torch.multinomial(probs, num_samples, replacement=True)

def SamplerQNN():
    return AdvancedSamplerQNN()

__all__ = ["SamplerQNN", "AdvancedSamplerQNN"]
