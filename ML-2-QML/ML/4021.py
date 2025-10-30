import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalSelfAttention(nn.Module):
    """Lightweight self‑attention block with learnable rotation and entanglement parameters."""
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.rotation_params = nn.Parameter(torch.randn(embed_dim * 3))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim - 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        rot = self.rotation_params.reshape(self.embed_dim, 3).cpu().numpy()
        ent = self.entangle_params.cpu().numpy()
        query = torch.matmul(inputs, torch.tensor(rot, dtype=torch.float32))
        key = torch.matmul(inputs, torch.tensor(ent.reshape(self.embed_dim, 1), dtype=torch.float32))
        scores = torch.softmax(query @ key.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ inputs).float()

class SamplerQNN(nn.Module):
    """Hybrid sampler: attention + feed‑forward sampler network."""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 4, embed_dim: int = 4):
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim=embed_dim)
        self.proj = nn.Linear(input_dim, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = self.proj(inputs)
        attn_out = self.attention(projected)
        out = self.net(attn_out)
        return F.softmax(out, dim=-1)

__all__ = ["SamplerQNN"]
