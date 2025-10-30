import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """
    Extended classical sampler network.
    • Two hidden layers with BatchNorm and Dropout for regularization.
    • Gumbel‑Softmax sampling for differentiable discrete output.
    """
    def __init__(self, input_dim: int = 2, hidden_dims: list[int] = [8, 6],
                 dropout: float = 0.2, temp: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], input_dim),
        )
        self.temp = temp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        # Gumbel‑Softmax to obtain differentiable samples
        return F.gumbel_softmax(logits, tau=self.temp, hard=True)
