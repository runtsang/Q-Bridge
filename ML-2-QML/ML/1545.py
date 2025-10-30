import torch
import torch.nn as nn
import numpy as np

class SelfAttention(nn.Module):
    """
    Extended classical self‑attention layer supporting batching,
    dropout, bias, and optional learnable rotation/entangle parameters.
    The public API mirrors the seed: run(rotation_params, entangle_params, inputs).
    """

    def __init__(self, embed_dim: int, dropout: float = 0.0, bias: bool = True, learn_params: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.bias = bias
        self.learn_params = learn_params
        if self.learn_params:
            # Learnable projections to generate attention parameters
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def run(self, rotation_params, entangle_params, inputs):
        """
        Compute self‑attention output.
        :param rotation_params: numpy array of shape (embed_dim, embed_dim)
        :param entangle_params: numpy array of shape (embed_dim,)
        :param inputs: torch tensor of shape (batch, seq_len, embed_dim)
        :return: torch tensor of shape (batch, seq_len, embed_dim)
        """
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)

        if self.learn_params:
            batch, seq_len, _ = inputs.shape
            flat = inputs.view(-1, self.embed_dim)
            rot = self.q_proj(flat).view(batch, seq_len, self.embed_dim, self.embed_dim)
            ent = self.k_proj(flat).view(batch, seq_len, self.embed_dim)
        else:
            rot = torch.from_numpy(rotation_params).float()
            ent = torch.from_numpy(entangle_params).float()

        Q = torch.einsum('bsi,ij->b sj', inputs, rot)
        K = torch.einsum('bsi,ij->b sj', inputs, ent)
        V = inputs
        scores = torch.softmax((Q @ K.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        scores = self.dropout(scores)
        out = torch.matmul(scores, V)
        return out
