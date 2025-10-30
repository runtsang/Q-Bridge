"""Hybrid classical self‑attention + QCNN model."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

__all__ = ["HybridSelfAttentionQCNN", "HybridAttentionQCNN"]


class ClassicalSelfAttention:
    """Light‑weight self‑attention using NumPy/PyTorch."""

    def __init__(self, embed_dim: int) -> None:
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Compute self‑attention scores and weighted sum."""
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


class QCNNModel(nn.Module):
    """QCNN‑style fully‑connected network."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


class HybridSelfAttentionQCNN(nn.Module):
    """
    Combines a classical self‑attention block with a QCNN stack.

    The attention block is applied element‑wise on each sample in the batch.
    The output is projected to the 8‑dimensional input expected by the
    QCNNModel. All learnable parameters are PyTorch tensors, enabling
    end‑to‑end gradient propagation.
    """

    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.attention = ClassicalSelfAttention(embed_dim)
        self.qcnn = QCNNModel()
        self.project = nn.Linear(embed_dim, 8) if embed_dim!= 8 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(batch, seq_len, embed_dim)``.

        Returns
        -------
        torch.Tensor
            Output of the QCNN stack after self‑attention preprocessing.
        """
        # Generate random parameters for the attention block; in practice
        # these could be learned or derived from a separate module.
        rotation_params = torch.randn(self.embed_dim * 3).numpy()
        entangle_params = torch.randn(self.embed_dim * 3).numpy()

        # Flatten batch and sequence dimensions for attention
        batch_size, seq_len, _ = x.shape
        x_flat = x.reshape(batch_size * seq_len, self.embed_dim)
        attn_out = self.attention.run(
            rotation_params, entangle_params, x_flat.numpy()
        )
        attn_out = torch.from_numpy(attn_out).reshape(batch_size, seq_len, self.embed_dim)

        # Project to 8‑dimensional feature space if needed
        if self.project is not None:
            attn_out = self.project(attn_out)

        # Collapse sequence dimension to feed the QCNN model
        qcnn_input = attn_out.reshape(batch_size, -1)
        return self.qcnn(qcnn_input)


def HybridAttentionQCNN() -> HybridSelfAttentionQCNN:
    """Factory returning a configured hybrid model."""
    return HybridSelfAttentionQCNN()
