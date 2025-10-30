import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNGen(nn.Module):
    """
    Extended classical sampler network with residual connections and dropout.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, output_dim: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return F.softmax(logits, dim=-1)
