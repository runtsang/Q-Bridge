import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention__gen181(nn.Module):
    """
    Hybrid classical model combining CNN feature extraction, classical self‑attention,
    a lightweight sampler network for classification, and a regression head.
    """

    def __init__(self, embed_dim: int = 4, num_classes: int = 4):
        super().__init__()
        # Feature extractor (Quantum‑NAT style)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_feat = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        self.norm_feat = nn.BatchNorm1d(embed_dim)

        # Self‑attention projections
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj   = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Sampler (classification head)
        self.sampler = nn.Sequential(
            nn.Linear(embed_dim, 8),
            nn.Tanh(),
            nn.Linear(8, num_classes)
        )

        # Estimator (regression head)
        self.estimator = nn.Sequential(
            nn.Linear(embed_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        f = self.features(x)
        f_flat = f.view(bsz, -1)
        feat = self.fc_feat(f_flat)
        return self.norm_feat(feat)

    def attention(self, h: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(h)
        k = self.key_proj(h)
        v = self.value_proj(h)
        scores = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.size(-1)), dim=-1)
        return torch.matmul(scores, v)

    def classify(self, h: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.sampler(h), dim=-1)

    def regress(self, h: torch.Tensor) -> torch.Tensor:
        return self.estimator(h)

    def forward(self, x: torch.Tensor):
        feat = self.forward_features(x)
        attn_out = self.attention(feat)
        logits = self.classify(attn_out)
        preds = self.regress(attn_out)
        return logits, preds
