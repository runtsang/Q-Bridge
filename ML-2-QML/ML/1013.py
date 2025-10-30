import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNEnhanced(nn.Module):
    """
    A deeper, regularised sampler network for 2‑dimensional inputs.
    Extends the original 2‑layer architecture with batch‑norm, dropout
    and an additional hidden layer, making it more expressive while
    still producing a 2‑class probability distribution.
    """

    def __init__(self, input_dim: int = 2, hidden_dims: list[int] = [8, 6], output_dim: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def init_weights(self) -> None:
        """Custom weight initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
