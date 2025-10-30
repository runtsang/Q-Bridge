import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HybridSelfAttention(nn.Module):
    """
    Classical hybrid self‑attention module that merges quanvolution style patch extraction,
    graph‑based adjacency weighting, and a lightweight estimator head.
    """
    def __init__(self,
                 embed_dim: int = 8,
                 patch_size: int = 2,
                 adjacency_threshold: float = 0.8,
                 estimator_hidden: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.adj_threshold = adjacency_threshold
        self.conv = nn.Conv2d(1, embed_dim, kernel_size=patch_size,
                              stride=patch_size, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.estimator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim * ((28 // patch_size) ** 2), estimator_hidden),
            nn.Tanh(),
            nn.Linear(estimator_hidden, 1)
        )

    def _adjacency_mask(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Build a binary adjacency mask from cosine similarities.
        """
        norms = embeddings.norm(dim=-1, keepdim=True) + 1e-12
        normed = embeddings / norms
        sims = torch.mm(normed, normed.t())
        mask = (sims >= self.adj_threshold).float()
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1, 28, 28)
        Returns: scalar predictions per batch
        """
        # Patch extraction
        patches = self.conv(x)  # (batch, embed_dim, H', W')
        batch, dim, h, w = patches.shape
        patches = patches.permute(0, 2, 3, 1).reshape(batch, -1, dim)  # (batch, N, dim)

        # Linear projections
        Q = self.q_proj(patches)
        K = self.k_proj(patches)
        V = self.v_proj(patches)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dim)
        scores = F.softmax(scores, dim=-1)

        # Adjacency weighting
        adjacency = self._adjacency_mask(patches.reshape(-1, dim))
        adjacency = adjacency.reshape(batch, -1, -1)
        weighted_scores = scores * adjacency

        # Weighted sum
        context = torch.matmul(weighted_scores, V)

        # Estimator head
        output = self.estimator(context)
        return output.squeeze(-1)
