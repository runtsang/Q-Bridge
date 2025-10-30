"""
Extended classical self‑attention with a learned projection and multi‑head capability.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Two‑stage, multi‑head self‑attention block.

    Parameters are supplied through a dictionary in the `run` method:
        - `proj_W`  : (embed_dim, proj_dim)
        - `proj_b`  : (proj_dim,)
        - `attn_Wq` : (proj_dim, num_heads, head_dim)
        - `attn_Wk` : (proj_dim, num_heads, head_dim)
        - `attn_Wv` : (proj_dim, num_heads, head_dim)
        - `out_W`   : (num_heads * head_dim, embed_dim)

    The block first projects the input into a lower‑dimensional space with a
    ReLU‑activated linear layer, then applies scaled dot‑product attention
    over `num_heads` heads, finally projecting back to `embed_dim`.
    """

    def __init__(self, embed_dim: int, proj_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.head_dim = proj_dim // num_heads

        # Placeholders – actual tensors are loaded in `run`
        self.proj_W = None
        self.proj_b = None
        self.attn_Wq = None
        self.attn_Wk = None
        self.attn_Wv = None
        self.out_W = None

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x @ self.proj_W + self.proj_b)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        # Shape: (B, L, num_heads, head_dim)
        q = x @ self.attn_Wq
        k = x @ self.attn_Wk
        v = x @ self.attn_Wv

        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, L, hd)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        heads = torch.matmul(weights, v)  # (B, heads, L, hd)
        heads = heads.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.head_dim)
        return heads @ self.out_W

    def run(self, inputs: np.ndarray, params: dict) -> np.ndarray:
        """
        Run the attention block.

        Args:
            inputs: (batch, seq_len, embed_dim) or (seq_len, embed_dim)
            params: dictionary with the weight tensors described above

        Returns:
            output: (batch, seq_len, embed_dim) or (seq_len, embed_dim)
        """
        if inputs.ndim == 2:
            inputs = inputs[None,...]  # add batch dimension

        x = torch.tensor(inputs, dtype=torch.float32)

        # Load parameters
        self.proj_W = torch.tensor(params["proj_W"], dtype=torch.float32)
        self.proj_b = torch.tensor(params["proj_b"], dtype=torch.float32)
        self.attn_Wq = torch.tensor(params["attn_Wq"], dtype=torch.float32)
        self.attn_Wk = torch.tensor(params["attn_Wk"], dtype=torch.float32)
        self.attn_Wv = torch.tensor(params["attn_Wv"], dtype=torch.float32)
        self.out_W = torch.tensor(params["out_W"], dtype=torch.float32)

        proj = self._project(x)
        out = self._attention(proj)
        # Remove batch dim if added
        return out.detach().numpy()[0] if inputs.shape[0] == 1 else out.detach().numpy()
