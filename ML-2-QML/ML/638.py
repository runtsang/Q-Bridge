import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """Multi‑head self‑attention with dropout and optional scaling.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default 4
        Number of attention heads.
    dropout : float, default 0.1
        Dropout probability applied to attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : Tensor of shape (batch, seq_len, embed_dim)

        Returns
        -------
        Tensor of shape (batch, seq_len, embed_dim)
        """
        batch, seq_len, _ = x.size()

        # Linear projections
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        out = self.out_proj(out)
        return out

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Compatibility wrapper that maps rotation_params to Q projection and
        entangle_params to K projection, mirroring the quantum interface.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters reshaped to (embed_dim, embed_dim) for Q projection.
        entangle_params : np.ndarray
            Parameters reshaped to (embed_dim, embed_dim) for K projection.
        inputs : np.ndarray
            Input embeddings of shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Output of the attention layer.
        """
        q_weight = rotation_params.reshape(self.embed_dim, -1)
        k_weight = entangle_params.reshape(self.embed_dim, -1)
        self.q_proj.weight.data = torch.from_numpy(q_weight.T).float()
        self.k_proj.weight.data = torch.from_numpy(k_weight.T).float()
        x = torch.from_numpy(inputs).float()
        return self.forward(x).detach().numpy()

__all__ = ["SelfAttention"]
