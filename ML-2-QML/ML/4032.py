import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridSelfAttentionQuanvolution(nn.Module):
    """Classical hybrid module that fuses quanvolution filtering with self‑attention."""
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        # quanvolution filter: 1 channel → embed_dim channels
        self.qfilter = nn.Conv2d(1, embed_dim, kernel_size=2, stride=2)
        # Attention linear projections
        self.out_proj = nn.Linear(embed_dim, 10)

    def run(self, inputs: torch.Tensor,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray) -> torch.Tensor:
        """
        Apply quanvolution, then self‑attention using the supplied parameters.
        :param inputs: batch of images, shape (B, 1, 28, 28)
        :param rotation_params: matrix for Q & K projections, shape (E, E)
        :param entangle_params: matrix for V projection, shape (E, E)
        :return: logits, shape (B, 10)
        """
        features = self.qfilter(inputs)  # (B, E, 14, 14)
        flat = features.view(features.size(0), -1)  # (B, E*14*14)
        D = flat.shape[1]
        # Convert parameters to tensors
        q_mat = torch.from_numpy(rotation_params).float().to(flat.device)
        k_mat = torch.from_numpy(entangle_params).float().to(flat.device)
        # Project to Q, K, V
        q = flat @ q_mat
        k = flat @ k_mat
        v = flat @ q_mat
        # Attention
        scores = torch.softmax(q @ k.T / np.sqrt(self.embed_dim), dim=-1)
        attn_out = scores @ v
        logits = self.out_proj(attn_out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridSelfAttentionQuanvolution"]
