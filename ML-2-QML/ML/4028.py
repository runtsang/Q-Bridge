import numpy as np
import torch
from torch import nn
from QCNN import QCNNModel

class SelfAttention:
    """
    Classical self‑attention that first passes the raw data through a QCNN
    feature extractor and then applies a standard scaled dot‑product
    attention.  The output dimension matches the embed_dim supplied at
    construction time and the attention weights are computed from
    rotation_params and entangle_params.
    """
    def __init__(self, embed_dim: int = 4):
        self.embed_dim = embed_dim
        # Use only the feature extraction part of QCNN
        qcnn = QCNNModel()
        self.feature_extractor = nn.Sequential(qcnn.feature_map, qcnn.conv1)

    def run(
        self,
        inputs: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        inputs : np.ndarray
            Raw input data of shape (batch, 8).  The QCNN feature extractor
            expects exactly 8‑dimensional feature vectors.
        rotation_params : np.ndarray
            Rotation angles of length embed_dim * 3.
        entangle_params : np.ndarray
            Entanglement angles of length embed_dim - 1.

        Returns
        -------
        np.ndarray
            Result of the attention operation.
        """
        # Feature extraction
        feats = self.feature_extractor(
            torch.as_tensor(inputs, dtype=torch.float32)
        ).detach()

        # Build query, key, value
        q = torch.matmul(
            feats,
            torch.as_tensor(rotation_params.reshape(self.embed_dim, -1),
                            dtype=torch.float32),
        )
        k = torch.matmul(
            feats,
            torch.as_tensor(entangle_params.reshape(self.embed_dim, -1),
                            dtype=torch.float32),
        )
        v = feats

        # Scaled dot‑product attention
        scores = torch.softmax(q @ k.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ v).numpy()

__all__ = ["SelfAttention"]
