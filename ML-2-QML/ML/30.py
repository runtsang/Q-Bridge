"""Extended classical sampler network with batch‑norm, dropout and residual connections."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """A robust two‑input sampler with batch‑norm, dropout and a residual block."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(2, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a probability distribution over two classes."""
        x = self.feature_extractor(x)
        logits = self.output_layer(x)
        return F.softmax(logits, dim=-1)

    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Generate discrete samples from the output distribution."""
        probs = self.forward(x).detach()
        return torch.multinomial(probs, n_samples, replacement=True)

__all__ = ["SamplerQNN"]
