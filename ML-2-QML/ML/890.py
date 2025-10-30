import torch
from torch import nn
import torch.nn.functional as F

class QCNNGen164(nn.Module):
    """Extended QCNN‑inspired architecture with residuals, batch‑norm and dropout."""
    def __init__(self, input_dim: int = 8, hidden_dims: list[int] | None = None, dropout: float = 0.2) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU()
        )
        self.layers = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                nn.BatchNorm1d(hidden_dims[i]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        # Residual skip to stabilize training
        self.residual = nn.Linear(hidden_dims[-1], hidden_dims[-1])
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.feature_map(x)
        for layer in self.layers:
            out = layer(out)
        out = out + self.residual(out)  # residual connection
        return torch.sigmoid(self.head(out))

def QCNNGen164() -> QCNNGen164:
    """Factory returning a configured QCNNGen164 instance."""
    return QCNNGen164()

__all__ = ["QCNNGen164", "QCNNGen164"]
