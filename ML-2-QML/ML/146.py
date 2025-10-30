import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """Classical neural sampler with extended architecture."""
    def __init__(self, input_dim: int = 2, hidden_dims: list[int] | None = None,
                 output_dim: int = 2, dropout: float = 0.2):
        super().__init__()
        hidden_dims = hidden_dims or [4, 8, 4]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities via softmax."""
        return F.softmax(self.net(x), dim=-1)

    def sample(self, n_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Generate samples from the learned probability distribution."""
        with torch.no_grad():
            # Use a zero‑vector to obtain the current distribution
            dummy = torch.zeros((n_samples, self.net[0].in_features), device=device)
            probs = self.forward(dummy)
            return torch.multinomial(probs, 1).squeeze(-1)

__all__ = ["SamplerQNN"]
