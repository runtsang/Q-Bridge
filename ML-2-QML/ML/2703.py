import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQCNNBinaryClassifier(nn.Module):
    """
    Classical approximation of the QCNN architecture.
    Mimics the feature‑map and convolutional pooling stages with
    fully‑connected layers, batch‑normalisation and dropout for
    regularisation.
    """
    def __init__(self, input_dim: int = 8) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.Tanh()
        )
        self.conv1 = nn.Sequential(
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.Tanh()
        )
        self.pool1 = nn.Sequential(
            nn.Linear(16, 12),
            nn.BatchNorm1d(12),
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.Linear(12, 8),
            nn.BatchNorm1d(8),
            nn.Tanh()
        )
        self.pool2 = nn.Sequential(
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.Tanh()
        )
        self.conv3 = nn.Sequential(
            nn.Linear(4, 4),
            nn.BatchNorm1d(4),
            nn.Tanh()
        )
        self.head = nn.Linear(4, 1)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        logits = self.head(x)
        probs = torch.sigmoid(logits)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridQCNNBinaryClassifier"]
