import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNGen162(nn.Module):
    """
    Extended sampler network with residual connections, dropout,
    and configurable hidden size for improved expressivity.
    """
    def __init__(self,
                 in_features: int = 2,
                 hidden_features: int = 8,
                 out_features: int = 2,
                 dropout: float = 0.1,
                 residual: bool = True) -> None:
        super().__init__()
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h2 = F.relu(self.fc2(h))
        if self.residual:
            h2 = h2 + h
        out = self.fc3(h2)
        return F.softmax(out, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        probs = self.forward(x).detach().cpu()
        return torch.multinomial(probs, n_samples, replacement=True)

__all__ = ["SamplerQNNGen162"]
