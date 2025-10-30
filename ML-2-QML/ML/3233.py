import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention:
    """
    Simple self‑attention helper that projects inputs using
    randomly generated rotation and entanglement parameters.
    """
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        # Reshape parameters to form square projection matrices
        rot = rotation_params.reshape(self.embed_dim, self.embed_dim)
        ent = entangle_params.reshape(self.embed_dim, self.embed_dim)
        # Compute query and key projections
        query = inputs @ rot
        key = inputs @ ent
        # Attention scores
        scores = np.exp(query @ key.T / np.sqrt(self.embed_dim))
        scores = scores / scores.sum(axis=-1, keepdims=True)
        # Weighted sum of values (original inputs)
        return scores @ inputs

class CombinedSamplerAttentionQNN(nn.Module):
    """
    Classical sampler network with an integrated self‑attention block.
    The attention re‑weights the input features before they are fed
    into the sampler network.
    """
    def __init__(self, embed_dim: int = 4, hidden_dim: int = 4):
        super().__init__()
        self.attention = SelfAttention(embed_dim)
        self.sampler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Convert to numpy for the attention module
        inp_np = inputs.detach().cpu().numpy()
        # Randomly initialise attention parameters
        rotation = np.random.rand(self.attention.embed_dim * self.attention.embed_dim)
        entangle = np.random.rand(self.attention.embed_dim * self.attention.embed_dim)
        # Apply attention
        attended = self.attention.run(rotation, entangle, inp_np)
        # Convert back to tensor
        attn_tensor = torch.as_tensor(attended, dtype=inputs.dtype, device=inputs.device)
        # Pass through sampler network
        logits = self.sampler(attn_tensor)
        return F.softmax(logits, dim=-1)

__all__ = ["CombinedSamplerAttentionQNN"]
