"""Hybrid Self‑Attention that fuses QCNN feature extraction with classical attention.

The class uses a lightweight QCNN model to transform raw inputs into
compact embeddings.  Those embeddings are then fed into a
self‑attention mechanism that is parameterised by
`rotation_params` and `entangle_params`.  The design mirrors the
original SelfAttention helper but now benefits from quantum‑style
feature learning.
"""

import numpy as np
import torch
from torch import nn

class QCNNModel(nn.Module):
    """Feature extractor inspired by the seed QCNN model.

    The forward pass returns a scalar, but the ``encode`` method exposes
    the intermediate feature map that can be used as the token
    representation for attention.
    """
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the 4‑dimensional embedding before the classification head."""
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return x

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.sigmoid(self.head(self.encode(inputs)))

class HybridSelfAttention:
    """Hybrid self‑attention combining QCNN embeddings with classical attention."""
    def __init__(self, embed_dim: int = 4, qcnn: QCNNModel | None = None) -> None:
        self.embed_dim = embed_dim
        self.qcnn = qcnn or QCNNModel()
        self.qcnn.eval()  # no‑gradient mode for inference

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute self‑attention using quantum‑inspired embeddings.

        Parameters
        ----------
        rotation_params : array of shape (embed_dim * 3,)
            Parameters for the per‑token rotation matrix.
        entangle_params : array of shape (embed_dim - 1,)
            Parameters for the pairwise entanglement.
        inputs : array of shape (seq_len, 8)
            Raw token vectors.

        Returns
        -------
        output : array of shape (seq_len, 4)
            The attended embeddings.
        """
        # Convert inputs to torch tensor
        inp = torch.as_tensor(inputs, dtype=torch.float32)

        # Obtain QCNN embeddings
        with torch.no_grad():
            feats = self.qcnn.encode(inp)  # shape (seq_len, 4)

        # Classical self‑attention on the embeddings
        query = torch.as_tensor(
            feats @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        key = torch.as_tensor(
            feats @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        value = feats
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

__all__ = ["HybridSelfAttention"]
