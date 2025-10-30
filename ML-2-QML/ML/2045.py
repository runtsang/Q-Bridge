import torch
from torch import nn
from typing import Optional

class QCNNModel(nn.Module):
    """
    Residual, dropout‑augmented fully‑connected network that mimics
    the structure of the quantum QCNN.  It can be used in any PyTorch
    training loop or as a feature extractor.
    """

    def __init__(self,
                 input_dim: int = 8,
                 hidden_dim: int = 16,
                 dropout: float = 0.1,
                 device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.pool1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.pool2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.pool3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(hidden_dim, 1)

        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)

        prev = x
        x1 = self.conv1(prev)
        x = self.pool1(self.dropout(x1 + prev))

        prev = x
        x2 = self.conv2(prev)
        x = self.pool2(self.dropout(x2 + prev))

        prev = x
        x3 = self.conv3(prev)
        x = self.pool3(self.dropout(x3))

        logits = self.head(x)
        return torch.sigmoid(logits)

def QCNN() -> QCNNModel:
    """
    Factory returning a configured :class:`QCNNModel`.
    """
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
