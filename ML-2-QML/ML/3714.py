import math
import torch
import torch.nn as nn

class QCNNModel(nn.Module):
    """Feed‑forward network mimicking the QCNN architecture."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class HybridSelfAttentionQCNN(nn.Module):
    """Hybrid self‑attention module built on top of a QCNN feature extractor."""
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.qcnn = QCNNModel()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of inputs of shape (batch, 8).

        Returns
        -------
        torch.Tensor
            Attention‑weighted representation of shape (batch, embed_dim).
        """
        features = self.qcnn(x)          # (batch, embed_dim)
        Q = self.query(features)
        K = self.key(features)
        V = self.value(features)
        scores = torch.softmax(Q @ K.transpose(-1, -2) / math.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, V)

__all__ = ["HybridSelfAttentionQCNN", "QCNNModel"]
