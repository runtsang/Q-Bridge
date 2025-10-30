"""Hybrid self‑attention module with multi‑head support and optional quantum refinement.

The implementation builds on the original SelfAttention helper but adds:
* multi‑head attention with configurable number of heads
* dropout regularisation
* GPU acceleration via PyTorch
* a lightweight factory for loading pretrained weight matrices
* a `run` convenience wrapper that accepts NumPy arrays for easy
  inter‑operation with the quantum counterpart.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

class SelfAttention__gen181(torch.nn.Module):
    """
    Classical self‑attention block.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default 1
        Number of attention heads.
    dropout : float, default 0.0
        Dropout probability applied to the attention weights.
    device : str or torch.device, default 'cpu'
        Target device.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = torch.nn.Dropout(dropout)
        self.device = torch.device(device)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute multi‑head self‑attention.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, seq_len, embed_dim).
        rotation_params : np.ndarray
            Shape (num_heads, embed_dim) – linear weights for Q.
        entangle_params : np.ndarray
            Shape (num_heads, embed_dim) – linear weights for K.
        return_attention : bool
            If True, also return the attention matrix.

        Returns
        -------
        output : torch.Tensor
            Shape (batch, seq_len, embed_dim).
        attn : torch.Tensor or None
            Attention weights per head (batch, num_heads, seq_len, seq_len).
        """
        # Convert parameters to torch tensors
        Q = torch.from_numpy(rotation_params).to(self.device).float()
        K = torch.from_numpy(entangle_params).to(self.device).float()

        B, T, _ = inputs.shape
        x = inputs.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)

        # Linear projections
        q = torch.einsum("bhid,hd->bhid", x, Q)  # (B, H, T, D)
        k = torch.einsum("bhid,hd->bhid", x, K)  # (B, H, T, D)
        v = x  # use original values as V

        # Scaled dot‑product attention
        attn_scores = torch.einsum("bhid,bhjd->bhij", q, k) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum
        out = torch.einsum("bhij,bhjd->bhid", attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)

        if return_attention:
            return out, attn_weights
        return out, None

    @staticmethod
    def from_pretrained(path: str, **kwargs) -> "SelfAttention__gen181":
        """
        Load a pretrained weight set from a.npz file containing
        `rotation_params` and `entangle_params` along with metadata.
        """
        data = np.load(path)
        embed_dim = int(data["embed_dim"])
        num_heads = int(data["num_heads"])
        rotation_params = data["rotation_params"]
        entangle_params = data["entangle_params"]
        model = SelfAttention__gen181(embed_dim, num_heads, **kwargs)
        # Store the parameters for later use
        model._rotation_params = rotation_params
        model._entangle_params = entangle_params
        return model

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        return_attention: bool = False,
    ) -> np.ndarray:
        """
        Compatibility wrapper that accepts NumPy arrays and returns NumPy.
        """
        inputs_t = torch.from_numpy(inputs).to(self.device).float()
        out_t, attn_t = self.forward(
            inputs_t, rotation_params, entangle_params, return_attention
        )
        out = out_t.cpu().detach().numpy()
        if return_attention:
            return out, attn_t.cpu().detach().numpy()
        return out

def SelfAttention() -> SelfAttention__gen181:
    """
    Factory mirroring the original seed's interface.
    """
    return SelfAttention__gen181(embed_dim=4)

__all__ = ["SelfAttention__gen181", "SelfAttention"]
