"""Hybrid self‑attention implemented in PyTorch.

The class mirrors the original SelfAttention interface but adds a
sampler network that learns a distribution over sequence positions.
This allows the classical attention to be modulated by a learned
probabilistic mask, providing a bridge to the quantum implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerModule(nn.Module):
    """
    Lightweight sampler that learns a probability distribution over the
    sequence length.  It is a direct PyTorch analogue of the QML SamplerQNN.
    """
    def __init__(self, seq_len: int, hidden_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, seq_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len)
        return F.softmax(self.net(x), dim=-1)

class HybridSelfAttention(nn.Module):
    """
    Classical self‑attention module with optional sampling modulation.
    Mimics the quantum SelfAttention interface by accepting rotation
    and entangle parameters, which are used to scale the projection
    matrices element‑wise.
    """
    def __init__(self, embed_dim: int, seq_len: int, use_sampler: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.use_sampler = use_sampler

        # Projection layers
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        if self.use_sampler:
            self.sampler = SamplerModule(seq_len)

    def forward(self, x: torch.Tensor,
                rotation_params: torch.Tensor | None = None,
                entangle_params: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input of shape (batch, seq_len, embed_dim)
        rotation_params, entangle_params : optional
            Mimic the quantum parameter vectors; if provided they are
            used to scale the projection matrices element‑wise.
        """
        # Scale projection weights if parameters are supplied
        if rotation_params is not None and rotation_params.shape[-1] == self.embed_dim:
            self.query_proj.weight.data *= rotation_params
        if entangle_params is not None and entangle_params.shape[-1] == self.embed_dim:
            self.key_proj.weight.data *= entangle_params

        query = self.query_proj(x)
        key   = self.key_proj(x)
        value = self.value_proj(x)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        if self.use_sampler:
            # Modulate attention with the sampler distribution
            sampler_weights = self.sampler(x.mean(dim=-1))   # (batch, seq_len)
            sampler_weights = sampler_weights.unsqueeze(-1)  # (batch, seq_len, 1)
            attn_weights = attn_weights * sampler_weights
            attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)

        out = torch.matmul(attn_weights, value)
        return out

__all__ = ["HybridSelfAttention"]
