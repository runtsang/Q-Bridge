import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNEnhanced(nn.Module):
    """
    A richer classical sampler network.
    Architecture:
        • Input projection 2 → 8
        • Residual block with two linear layers + BatchNorm
        • Dropout for regularisation
        • Output layer 2 with log‑softmax
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, output_dim: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_layer(x))
        h = self.res_block(h) + h  # residual connection
        h = self.dropout(h)
        logits = self.output_layer(h)
        return F.log_softmax(logits, dim=-1)
