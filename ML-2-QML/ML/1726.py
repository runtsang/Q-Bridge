import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNGen310(nn.Module):
    """Extended sampler network with batch normalization, dropout, and two hidden layers.
    Designed to better approximate categorical distributions for downstream hybrid models.
    """
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 8,
                 output_dim: int = 2,
                 dropout_prob: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def sample(self,
               x: torch.Tensor,
               num_samples: int = 1) -> torch.Tensor:
        """Return hard samples drawn from the output probability distribution."""
        probs = self.forward(x)
        return torch.multinomial(probs, num_samples, replacement=True).squeeze(-1)

__all__ = ["SamplerQNNGen310"]
