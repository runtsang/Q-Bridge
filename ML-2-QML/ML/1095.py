import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNGen(nn.Module):
    """
    A robust classical sampler network with dropout and batch normalization.
    The network outputs a probability distribution over two classes.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, output_dim: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Sample from the categorical distribution defined by the network output.
        """
        probs = self.forward(x)
        return torch.multinomial(probs, num_samples, replacement=True)
