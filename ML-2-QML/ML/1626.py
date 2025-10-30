import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """
    A robust classical sampler network that extends the original two‑layer
    architecture.  It contains two hidden layers, batch‑norm, residual
    connections, dropout and a temperature parameter that controls the
    sharpness of the output distribution.
    """

    def __init__(self, in_features: int = 2, hidden: int = 8, out_features: int = 2,
                 dropout: float = 0.2, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.out = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns a temperature‑scaled softmax probability
        distribution over ``out_features``.
        """
        h = self.net(x)
        logits = self.out(h)
        logits = logits / self.temperature
        return F.softmax(logits, dim=-1)

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sample from the probability distribution produced by :meth:`forward`.
        Returns integer class indices.
        """
        probs = self.forward(x)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

__all__ = ["SamplerQNN"]
