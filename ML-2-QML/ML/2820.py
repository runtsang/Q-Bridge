import torch
import torch.nn as nn
import numpy as np

class HybridSelfAttention(nn.Module):
    """
    Classical self‑attention module that incorporates a lightweight neural
    estimator (mimicking the quantum EstimatorQNN) to refine attention scores.
    The architecture mirrors the original SelfAttention.py but replaces the
    quantum block with a small feed‑forward network.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        # Projection layers for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # Estimator sub‑network: two‑layer MLP producing a scalar per key‑value pair
        self.estimator = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Raw dot‑product scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.embed_dim)

        # Refine scores with estimator: concatenate Q and K for each pair
        batch, seq_len, _ = Q.shape
        q_flat = Q.reshape(batch, seq_len, 1, self.embed_dim)
        k_flat = K.reshape(batch, 1, seq_len, self.embed_dim)
        pair_features = torch.cat(
            [q_flat.expand(-1, -1, seq_len, -1),
             k_flat.expand(-1, seq_len, -1, -1)],
            dim=-1,
        )  # shape (batch, seq_len, seq_len, 2*embed_dim)
        estimator_scores = self.estimator(pair_features.reshape(-1, 2 * self.embed_dim))
        estimator_scores = estimator_scores.reshape(batch, seq_len, seq_len)

        # Combine raw and estimator scores
        combined_scores = scores + estimator_scores
        attn_weights = torch.softmax(combined_scores, dim=-1)
        return torch.matmul(attn_weights, V)

__all__ = ["HybridSelfAttention"]
