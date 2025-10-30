import torch
from torch import nn
import numpy as np

class QCSelfAttentionModel(nn.Module):
    """
    A classical QCNN model enhanced with a self‑attention block.
    The architecture follows the original QCNN layers and adds a
    Multi‑Head Attention after the final convolution to capture
    long‑range dependencies in the feature map.
    """
    def __init__(self, embed_dim: int = 4, num_heads: int = 1):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Self‑attention operates on the 4‑dimensional vector
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim,
                                               num_heads=num_heads,
                                               batch_first=True)
        self.head = nn.Linear(4, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 8).

        Returns
        -------
        torch.Tensor
            Output probabilities of shape (batch, 1).
        """
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        # Attention expects (batch, seq_len, embed_dim)
        attn_out, _ = self.attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x = attn_out.squeeze(1)
        x = self.dropout(x)
        return torch.sigmoid(self.head(x))

def QCNNSelfAttention() -> QCSelfAttentionModel:
    """
    Factory returning a configured QCSelfAttentionModel.
    """
    return QCSelfAttentionModel()
