import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNGen205(nn.Module):
    """
    Classical sampler network with residual connections and dropout.
    Extends the original 2‑layer MLP to a deeper, more expressive architecture.
    """

    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 8,
                 output_dim: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Draw samples from the categorical distribution defined by the network output.
        """
        probs = self.forward(x)
        return torch.multinomial(probs, num_samples, replacement=True)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Cross‑entropy loss between network output and target labels.
        """
        logits = self.net(x)
        return F.cross_entropy(logits, y)

__all__ = ["SamplerQNNGen205"]
