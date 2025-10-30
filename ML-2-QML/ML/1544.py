import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """
    Extended classical sampler network.
    Accepts a 2‑dimensional input and outputs a 2‑class probability distribution.
    The architecture comprises three hidden layers with LeakyReLU activations,
    dropout and weight decay, providing richer expressivity while remaining lightweight.
    """
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(16, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns a probability distribution over the 2 output classes.
        """
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Sample from the output distribution.
        """
        probs = self.forward(x)
        probs_expanded = probs.unsqueeze(1).repeat(1, num_samples, 1)
        samples = torch.multinomial(probs_expanded.reshape(-1, 2), 1).reshape(probs.shape[0], num_samples)
        return samples

__all__ = ["SamplerQNN"]
