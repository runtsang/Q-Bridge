import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttentionModel(nn.Module):
    """
    Enhanced multi‑head self‑attention module with bias and dropout.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    heads : int, default 4
        Number of attention heads.
    dropout : float, default 0.1
        Dropout probability applied to the attention weights.
    """

    def __init__(self, embed_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        assert embed_dim % heads == 0, "embed_dim must be divisible by heads"

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                inputs: torch.Tensor,
                rotation_params: np.ndarray = None,
                entangle_params: np.ndarray = None) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, seq_len, embed_dim).
        rotation_params : np.ndarray, optional
            Weight matrix reshaped to (embed_dim, embed_dim) used to
            initialise the linear projections.
        entangle_params : np.ndarray, optional
            Scaling vector of length ``seq_len`` applied to the raw
            attention logits before the softmax.
        """
        if rotation_params is not None:
            weight = torch.as_tensor(rotation_params, dtype=torch.float32)
            self.q_proj.weight.data.copy_(weight[:self.embed_dim, :self.embed_dim])
            self.k_proj.weight.data.copy_(weight[self.embed_dim:2 * self.embed_dim, :self.embed_dim])
            self.v_proj.weight.data.copy_(weight[2 * self.embed_dim:, :self.embed_dim])

        if entangle_params is not None:
            scale = torch.as_tensor(entangle_params, dtype=torch.float32)
            scale = scale.unsqueeze(0).unsqueeze(-1)  # broadcast over batch, heads

        # Linear projections
        Q = self.q_proj(inputs)
        K = self.k_proj(inputs)
        V = self.v_proj(inputs)

        # Split heads
        B, S, _ = Q.shape
        Q = Q.view(B, S, self.heads, self.head_dim).transpose(1, 2)  # (B, H, S, D)
        K = K.view(B, S, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        if entangle_params is not None:
            scores = scores * scale

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        out = self.out_proj(out)
        return out

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Convenience wrapper matching the original interface.
        """
        inputs_t = torch.as_tensor(inputs, dtype=torch.float32)
        out_t = self.forward(inputs_t, rotation_params, entangle_params)
        return out_t.detach().numpy()
