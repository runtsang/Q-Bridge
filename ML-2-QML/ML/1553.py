import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Depthwise‑like residual block with channel‑wise convolution and skip connection."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn(self.conv(x)))
        out = self.dropout(out)
        if x.shape[1]!= out.shape[1]:
            x = F.pad(x, (0, 0, 0, out.shape[1] - x.shape[1]))
        return F.relu(out + x)

class QCNNModel(nn.Module):
    """Classical QCNN inspired by quantum convolution layers with residuals and pooling."""
    def __init__(self, in_features: int = 8, hidden_channels: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_features, hidden_channels)
        self.layer1 = ResidualBlock(hidden_channels, hidden_channels, kernel_size=3, dropout=dropout)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.layer2 = ResidualBlock(hidden_channels, hidden_channels, kernel_size=3, dropout=dropout)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.layer3 = ResidualBlock(hidden_channels, hidden_channels, kernel_size=3, dropout=dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x).unsqueeze(-1)
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.layer3(x)
        x = self.global_pool(x).squeeze(-1)
        return torch.sigmoid(self.head(x))

def QCNN() -> QCNNModel:
    """Factory returning a fully‑initialized QCNNModel."""
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
