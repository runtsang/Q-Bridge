import numpy as np
import torch
from torch import nn
from typing import Tuple

# ------------------------------------------------------------
# Classical QCNN feature extractor (inspired by the seed)
# ------------------------------------------------------------
class QCNNModel(nn.Module):
    """Stack of linear layers emulating a QCNN feature extractor."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim // 2), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(hidden_dim // 4, hidden_dim // 4), nn.Tanh())
        self.head = nn.Linear(hidden_dim // 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

def QCNN() -> QCNNModel:
    """Factory returning the QCNN feature extractor."""
    return QCNNModel()

# ------------------------------------------------------------
# Classical self‑attention block that consumes QCNN features
# ------------------------------------------------------------
class SelfAttentionQCNN:
    """Hybrid classical model: QCNN feature extractor + multi‑head attention."""
    def __init__(self, embed_dim: int = 4, num_heads: int = 2):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qcnn = QCNN()
        self.scale = np.sqrt(self.embed_dim)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the linear projections (Q, K, V). Shape:
            (num_heads, embed_dim, embed_dim)
        entangle_params : np.ndarray
            Not used in the classical branch but kept for API parity.
        inputs : np.ndarray
            Shape (batch, feature_dim)
        Returns
        -------
        np.ndarray
            Attention‑weighted feature representation.
        """
        # 1. QCNN feature extraction
        with torch.no_grad():
            torch_inputs = torch.as_tensor(inputs, dtype=torch.float32)
            features = self.qcnn(torch_inputs).numpy()

        # 2. Multi‑head attention
        batch, dim = features.shape
        queries = torch.as_tensor(
            features @ rotation_params[:, 0, :].reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        keys = torch.as_tensor(
            features @ rotation_params[:, 1, :].reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        values = torch.as_tensor(features, dtype=torch.float32)

        scores = torch.softmax(
            queries @ keys.T / self.scale, dim=-1
        )
        output = (scores @ values).numpy()
        return output

def SelfAttention() -> SelfAttentionQCNN:
    """Factory returning the hybrid self‑attention model."""
    return SelfAttentionQCNN()

__all__ = ["SelfAttention", "SelfAttentionQCNN", "QCNN", "QCNNModel"]
