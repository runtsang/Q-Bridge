from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Classical multi‑head self‑attention with optional regression head
# --------------------------------------------------------------------------- #
class ClassicalSelfAttentionHybrid(nn.Module):
    """
    Multi‑head attention that accepts external rotation and entanglement
    parameters to mimic the quantum interface.  The parameters are treated
    as per‑head scaling factors applied after the linear projections.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        rotation_params: np.ndarray | None,
        entangle_params: np.ndarray | None,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (num_heads, head_dim).  Acts as a multiplicative factor on Q.
        entangle_params : np.ndarray
            Shape (num_heads, head_dim).  Acts as a multiplicative factor on K.
        inputs : torch.Tensor
            Shape (batch, seq_len, embed_dim).
        """
        batch, seq_len, _ = inputs.shape

        # Linear projections
        Q = self.q_proj(inputs)
        K = self.k_proj(inputs)
        V = self.v_proj(inputs)

        # Reshape for multi‑head
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply external parameters if provided
        if rotation_params is not None:
            Q = Q * torch.tensor(rotation_params, device=inputs.device).unsqueeze(0).unsqueeze(2)
        if entangle_params is not None:
            K = K * torch.tensor(entangle_params, device=inputs.device).unsqueeze(0).unsqueeze(2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)

        # Weighted sum
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        out = self.out_proj(out)
        return out

# --------------------------------------------------------------------------- #
# Dataset & regression head
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Classic superposition dataset used for regression experiments.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Simple PyTorch dataset wrapping the superposition data.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QModel(nn.Module):
    """
    End‑to‑end regression model that stacks the hybrid attention block
    followed by a small feed‑forward head.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.attn = ClassicalSelfAttentionHybrid(embed_dim=num_features, num_heads=4)
        self.regressor = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ignore external params for the demo; pass zeros
        rotation = np.zeros((self.attn.num_heads, self.attn.head_dim))
        entangle = np.zeros_like(rotation)
        attn_out = self.attn(rotation, entangle, x)
        # Aggregate over sequence dimension
        pooled = attn_out.mean(dim=1)
        return self.regressor(pooled).squeeze(-1)

__all__ = ["ClassicalSelfAttentionHybrid", "RegressionDataset", "QModel", "generate_superposition_data"]
