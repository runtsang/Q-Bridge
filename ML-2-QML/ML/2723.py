import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention:
    """Simple self‑attention that operates on NumPy arrays."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key   = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

class HybridSamplerAttention(nn.Module):
    """
    Classical sampler that first applies a learnable self‑attention block
    and then maps the attended representation to a 2‑class probability
    distribution via a small feed‑forward network.
    """
    def __init__(self, input_dim: int = 2, embed_dim: int = 4):
        super().__init__()
        # Learnable parameters that mimic the quantum rotation/entanglement
        self.rotation_params = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim))
        self.attention = SelfAttention(embed_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Compute attention output on the raw inputs
        attn_out = self.attention.run(
            rotation_params=self.rotation_params.detach().numpy(),
            entangle_params=self.entangle_params.detach().numpy(),
            inputs=inputs.detach().numpy(),
        )
        # Convert back to a torch tensor for the feed‑forward part
        combined = torch.from_numpy(attn_out).to(inputs.device)
        logits = self.net(combined)
        return F.softmax(logits, dim=-1)

__all__ = ["HybridSamplerAttention"]
