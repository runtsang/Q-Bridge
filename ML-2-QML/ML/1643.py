import torch
import torch.nn as nn
import torch.nn.functional as F

class QFCModel(nn.Module):
    """Extended CNN with residual connections, dropout, and 4‑feature output."""
    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Residual block
        self.residual = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
        )
        self.bn_res = nn.BatchNorm2d(16)
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        res = self.residual(x)
        x = F.relu(x + res)
        x = self.bn_res(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.norm(x)
