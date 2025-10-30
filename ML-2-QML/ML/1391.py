import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSamplerQNN(nn.Module):
    """
    A deeper classical sampler network with residual connections and dropout.
    Improves expressivity over the original 2‑layer architecture while
    maintaining a simple interface for probability output.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, output_dim: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        residual = x
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = x + residual  # residual connection
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

__all__ = ["HybridSamplerQNN"]
