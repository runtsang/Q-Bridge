import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ClassicalSelfAttention:
    """A lightweight self‑attention module that mimics the quantum‑style interface."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        query = np.dot(inputs, rotation_params.reshape(self.embed_dim, -1))
        key   = np.dot(inputs, entangle_params.reshape(self.embed_dim, -1))
        scores = np.exp(query @ key.T / np.sqrt(self.embed_dim))
        scores /= scores.sum(axis=-1, keepdims=True)
        return scores @ inputs

class QCNNGen144(nn.Module):
    """Classical hybrid‑inspired QCNN with attention and sampler‑style output."""
    def __init__(self, input_dim: int = 8, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Linear(16, 12), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Linear(12, 8), nn.ReLU())
        self.attn = ClassicalSelfAttention(embed_dim)
        self.final_net = nn.Sequential(
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # Use random attention weights for demonstration; in practice trainable params would be used
        rot = np.random.randn(self.embed_dim, x.shape[-1])
        ent = np.random.randn(self.embed_dim, x.shape[-1])
        attn_out = self.attn.run(rot, ent, x.detach().cpu().numpy())
        x = torch.from_numpy(attn_out).to(x.device)
        out = self.final_net(x)
        return torch.sigmoid(out).squeeze(-1)

__all__ = ["QCNNGen144"]
