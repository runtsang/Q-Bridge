import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """A configurable classical sampler network."""
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 8,
        output_dim: int = 2,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

def SamplerQNN_factory(**kwargs) -> SamplerQNN:
    """Convenience factory that returns an instance of SamplerQNN."""
    return SamplerQNN(**kwargs)

__all__ = ["SamplerQNN", "SamplerQNN_factory"]
