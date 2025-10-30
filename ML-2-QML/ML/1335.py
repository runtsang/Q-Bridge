import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """
    A lightweight yet expressive classical sampler network.
    Extends the original 2‑input, 2‑output architecture by adding
    batch‑norm, dropout and a second hidden layer.
    """
    def __init__(self, hidden_dim: int = 32, dropout: float = 0.1, seed: int | None = None) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)

    def sample(self, n: int = 1) -> torch.Tensor:
        """
        Draw samples from the categorical distribution produced by the network.
        """
        probs = self.forward(torch.randn(n, 2))
        return torch.multinomial(probs, 1).squeeze(-1)

__all__ = ["SamplerQNN"]
