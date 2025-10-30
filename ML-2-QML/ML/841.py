import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQFCModel(nn.Module):
    """Extended CNN with residual connections and dropout for 4‑feature output."""
    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.residual = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 4)
        )
        self.out_norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        res = self.residual(x1)
        x = x2 + res
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.out_norm(x)

__all__ = ["HybridQFCModel"]
