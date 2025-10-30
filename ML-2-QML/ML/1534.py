import torch
from torch import nn
import torch.nn.functional as F

class QCNN(nn.Module):
    """
    A QCNN‑inspired fully‑connected network with modern regularisation.
    Mirrors the original architecture but adds batch‑norm, ReLU,
    dropout and He weight initialization to stabilise training.
    """
    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16, bias=True),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.conv1 = nn.Sequential(
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.pool1 = nn.Sequential(
            nn.Linear(16, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Linear(12, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.pool2 = nn.Sequential(
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Linear(4, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.head = nn.Linear(4, 1, bias=True)

        # He (Kaiming) uniform initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        logits = self.head(x)
        return torch.sigmoid(logits)

__all__ = ["QCNN"]
