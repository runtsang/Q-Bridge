import torch
from torch import nn
import numpy as np

# Classical self‑attention helper (mirrors the ML seed)
class SelfAttention:
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1),
                                dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1),
                              dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

# Hybrid classical regressor
class HybridEstimatorQNN(nn.Module):
    """
    Combines a feed‑forward regression head with a self‑attention module.
    The attention module learns to weight the input features before they
    are passed to the regression network.
    """
    def __init__(self, input_dim: int = 2, embed_dim: int = 4):
        super().__init__()
        self.attention = SelfAttention(embed_dim=embed_dim)
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Random parameters for the attention mechanism (could be learnable)
        rot = torch.randn(self.attention.embed_dim, dtype=torch.float32)
        ent = torch.randn(self.attention.embed_dim, dtype=torch.float32)
        # Compute attention‑weighted features
        attn_features = self.attention.run(
            rotation_params=rot.numpy(),
            entangle_params=ent.numpy(),
            inputs=x.numpy(),
        )
        attn_tensor = torch.from_numpy(attn_features)
        return self.regressor(attn_tensor)

__all__ = ["HybridEstimatorQNN"]
