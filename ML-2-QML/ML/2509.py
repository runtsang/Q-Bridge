import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttentionHybrid(nn.Module):
    """
    Classical hybrid of CNN, self‑attention and optional quantum attention.
    """
    def __init__(self, use_quantum_attention: bool = False):
        super().__init__()
        self.use_quantum_attention = use_quantum_attention

        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Self‑attention projections
        self.query_proj = nn.Linear(16 * 7 * 7, 64)
        self.key_proj   = nn.Linear(16 * 7 * 7, 64)
        self.value_proj = nn.Linear(16 * 7 * 7, 64)

        # Classification head
        self.fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)

    def classical_attention(self, features: torch.Tensor) -> torch.Tensor:
        bsz, c, h, w = features.shape
        flat = features.view(bsz, -1)
        Q = self.query_proj(flat)
        K = self.key_proj(flat)
        V = self.value_proj(flat)
        scores = F.softmax(Q @ K.t() / np.sqrt(Q.size(-1)), dim=-1)
        return scores @ V

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        if self.use_quantum_attention:
            # Placeholder for quantum attention integration
            attn = feat.view(bsz, -1)
        else:
            attn = self.classical_attention(feat)
        out = self.fc(attn)
        return self.norm(out)

__all__ = ["SelfAttentionHybrid"]
