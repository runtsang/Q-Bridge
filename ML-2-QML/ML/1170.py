import torch
from torch import nn
from torch.nn import functional as F

class QCNNModel(nn.Module):
    """
    A deep residual QCNN‑inspired architecture with batch‑norm, dropout and
    flexible feature extraction.  The design mirrors the original
    fully‑connected stack but adds residual connections and regularisation
    to improve generalisation on small datasets.
    """
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16, num_classes: int = 1) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Residual block 1
        self.res1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        # Residual block 2
        self.res2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        # Pooling stages
        self.pool1 = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * 0.75)),
            nn.ReLU()
        )
        self.pool2 = nn.Sequential(
            nn.Linear(int(hidden_dim * 0.75), int(hidden_dim * 0.5)),
            nn.ReLU()
        )

        # Classification head
        self.head = nn.Linear(int(hidden_dim * 0.5), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        # Residual connections
        res = self.res1(x)
        x = F.relu(x + res)
        res = self.res2(x)
        x = F.relu(x + res)
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.head(x)
        return torch.sigmoid(x) if self.head.out_features == 1 else x

def QCNN() -> QCNNModel:
    """Return a default QCNNModel instance."""
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
