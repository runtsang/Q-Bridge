import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    """
    Multi‑head self‑attention block with residual connections.
    """

    def __init__(self, embed_dim: int, head_count: int = 4, depth: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_count = head_count
        self.depth = depth
        if embed_dim % head_count!= 0:
            raise ValueError("embed_dim must be divisible by head_count")

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray):
        """
        Forward pass compatible with the original interface.

        Parameters
        ----------
        rotation_params : np.ndarray
            Flat array of shape (embed_dim * embed_dim,) used for Q/K projections.
        entangle_params : np.ndarray
            Flat array of shape (embed_dim * embed_dim,) used for V projection.
        inputs : np.ndarray
            Input tensor of shape (..., embed_dim).

        Returns
        -------
        np.ndarray
            Transformed tensor of the same shape as `inputs`.
        """
        x = torch.from_numpy(inputs).float()
        q_weight = torch.from_numpy(rotation_params.reshape(self.embed_dim, self.embed_dim))
        k_weight = torch.from_numpy(entangle_params.reshape(self.embed_dim, self.embed_dim))
        v_weight = torch.from_numpy(entangle_params.reshape(self.embed_dim, self.embed_dim))

        for _ in range(self.depth):
            # Linear projections
            q = x @ q_weight  # (..., embed_dim)
            k = x @ k_weight
            v = x @ v_weight

            # Split into heads
            q = q.view(*q.shape[:-1], self.head_count, self.embed_dim // self.head_count)
            k = k.view(*k.shape[:-1], self.head_count, self.embed_dim // self.head_count)
            v = v.view(*v.shape[:-1], self.head_count, self.embed_dim // self.head_count)

            # Scaled dot‑product attention per head
            scores = torch.einsum("...hqd,...hkd->...hqk", q, k) / np.sqrt(self.embed_dim // self.head_count)
            attn = F.softmax(scores, dim=-1)

            # Weighted sum of values
            out = torch.einsum("...hqk,...hvd->...hqd", attn, v)

            # Concatenate heads
            out = out.reshape(*out.shape[:-2], self.embed_dim)

            # Residual connection
            x = out + x

        return x.detach().cpu().numpy()

__all__ = ["SelfAttention"]
