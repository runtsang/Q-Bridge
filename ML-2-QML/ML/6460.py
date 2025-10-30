import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalSelfAttention:
    """Simple parametric self‑attention module used in the classical branch."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key   = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

def SelfAttention():
    """Factory returning a ClassicalSelfAttention instance."""
    return ClassicalSelfAttention(embed_dim=4)

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 convolution that emulates a quanvolution filter."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class QuanvolutionSelfAttentionClassifier(nn.Module):
    """Hybrid classifier that combines a quanvolution filter with a self‑attention head."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.attention = SelfAttention()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract 2×2 patches via a classical convolution
        features = self.qfilter(x)                # (batch, 4*14*14)
        batch = features.shape[0]
        seq_len = 14 * 14
        embed_dim = 4
        features = features.view(batch, seq_len, embed_dim)

        # Compute a global mean embedding per sample
        mean_embeds = features.mean(dim=1).cpu().detach().numpy()  # (batch, embed_dim)

        # Random attention parameters for each forward pass
        rotation_params = np.random.rand(embed_dim * embed_dim)
        entangle_params = np.random.rand(embed_dim * embed_dim)

        # Apply attention to each sample
        attn_weights = []
        for embed in mean_embeds:
            # Pass a 1‑D embedding as a 2‑D array to match the API
            weights = self.attention.run(rotation_params, entangle_params, embed[np.newaxis, :])
            attn_weights.append(weights.squeeze())
        attn_weights = torch.tensor(attn_weights, device=features.device, dtype=torch.float32)

        # Modulate patch embeddings with the learned attention weights
        features = features * attn_weights.unsqueeze(1)  # (batch, seq_len, embed_dim)
        features = features.view(batch, -1)              # (batch, 4*14*14)

        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionSelfAttentionClassifier"]
