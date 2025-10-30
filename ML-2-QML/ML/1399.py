import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATEnhanced(nn.Module):
    """Classical CNN with residual skip, dropout and a learnable scaling factor."""
    def __init__(self, dropout: float = 0.3, scaling_init: float = 1.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)
        # Learnable scaling applied after batch‑norm
        self.scaling = nn.Parameter(torch.full((1,), scaling_init))
        # Residual projection to match output dimensionality
        self.res_proj = nn.Linear(16 * 7 * 7, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        out = self.fc(self.dropout(flat))
        # Residual connection
        out = out + self.res_proj(flat)
        out = self.norm(out)
        out = out * self.scaling
        return out

__all__ = ["QuantumNATEnhanced"]
