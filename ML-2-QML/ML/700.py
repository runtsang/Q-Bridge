import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNExtended(nn.Module):
    """
    A deeper, regularised sampler network.
    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input.
    hidden_dim : int, default 8
        Size of hidden layers.
    output_dim : int, default 2
        Number of output classes.
    dropout : float, default 0.1
        Dropout probability applied after the first hidden layer.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8,
                 output_dim: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        return F.softmax(self.net(x), dim=-1)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross‑entropy loss."""
        return F.cross_entropy(logits, targets)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return argmax class predictions."""
        probs = self.forward(x)
        return probs.argmax(dim=1)

__all__ = ["SamplerQNNExtended"]
