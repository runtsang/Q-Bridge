"""
SelfAttention__gen038.py (ML portion)

This module implements a classical self‑attention block that internally uses a
QCNN‑style fully‑connected network as a feature extractor.  It preserves the
`run` signature of the original `SelfAttention` seed while adding a deeper
representation and a configurable attention dimension.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# 1.  Classical QCNN feature extractor
# --------------------------------------------------------------------------- #
class QCNNFeatureExtractor(nn.Module):
    """
    A lightweight QCNN‑inspired network that transforms raw inputs into
    a richer embedding.  The architecture is deliberately shallow so that
    it can be composed easily with the self‑attention block.
    """
    def __init__(self, in_features: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

# --------------------------------------------------------------------------- #
# 2.  Self‑attention block
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    """
    Classical self‑attention that accepts rotation and entangle parameters
    purely for API compatibility.  The parameters are ignored by the forward
    path, but are stored for introspection and potential future learning.
    """
    def __init__(self, embed_dim: int, feature_dim: int | None = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim or embed_dim
        self.feature_extractor = QCNNFeatureExtractor(self.feature_dim)
        # Linear maps that emulate the rotation/entangle transforms
        self.query_proj = nn.Linear(self.feature_dim, self.embed_dim)
        self.key_proj   = nn.Linear(self.feature_dim, self.embed_dim)
        self.value_proj = nn.Linear(self.feature_dim, self.embed_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: np.ndarray | None = None,
        entangle_params: np.ndarray | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Raw input of shape (batch, features).
        rotation_params, entangle_params : np.ndarray, optional
            Accepted for API parity; ignored in the forward pass.

        Returns
        -------
        torch.Tensor
            The attention‑weighted output of shape (batch, embed_dim).
        """
        # Feature extraction
        features = self.feature_extractor(inputs)
        # Linear projections
        query = self.query_proj(features)
        key   = self.key_proj(features)
        value = self.value_proj(features)
        # Attention scores
        scores = torch.softmax(
            torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(self.embed_dim),
            dim=-1,
        )
        return torch.matmul(scores, value)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compatibility wrapper that mirrors the original seed's run interface.
        """
        tensor_inputs = torch.as_tensor(inputs, dtype=torch.float32)
        output = self.forward(tensor_inputs, rotation_params, entangle_params)
        return output.detach().cpu().numpy()

# --------------------------------------------------------------------------- #
# 3.  Factory
# --------------------------------------------------------------------------- #
def SelfAttention() -> ClassicalSelfAttention:
    """
    Factory that returns a pre‑configured ClassicalSelfAttention instance.
    """
    return ClassicalSelfAttention(embed_dim=4, feature_dim=8)

__all__ = ["SelfAttention", "ClassicalSelfAttention", "QCNNFeatureExtractor"]
