import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerModule(nn.Module):
    """Enhanced classical sampler network.

    Features:
    - 3 hidden layers with GELU activation.
    - Optional dropout for regularisation.
    - Log‑softmax output for numerical stability.
    """
    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(8, 8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(8, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return log‑probabilities over 2 classes."""
        return F.log_softmax(self.net(inputs), dim=-1)

def SamplerQNN(dropout: float = 0.0) -> nn.Module:
    """Factory for SamplerModule with optional dropout."""
    return SamplerModule(dropout=dropout)

__all__ = ["SamplerQNN"]
