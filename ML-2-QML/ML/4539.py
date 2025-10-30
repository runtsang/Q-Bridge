import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalQuanvolutionFilter(nn.Module):
    """Classical 2×2 patch encoder mimicking the quantum quanvolution."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class RBFKernelLayer(nn.Module):
    """Trainable RBF mapping to a set of reference vectors."""
    def __init__(self, input_dim: int, num_refs: int, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.ref_vectors = nn.Parameter(torch.randn(num_refs, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        diff = x.unsqueeze(1) - self.ref_vectors.unsqueeze(0)  # (batch, num_refs, input_dim)
        dist_sq = (diff ** 2).sum(dim=2)  # (batch, num_refs)
        return torch.exp(-self.gamma * dist_sq)

class SelfAttentionBlock(nn.Module):
    """Simple self‑attention on a 1‑D feature vector."""
    def __init__(self, embed_dim: int, num_heads: int = 1) -> None:
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(q.size(-1)), dim=-1)
        return scores @ v

class HybridQuantumNat(nn.Module):
    """Classical hybrid model combining CNN, quanvolution, kernel, and self‑attention."""
    def __init__(self, num_classes: int = 10, gamma: float = 1.0) -> None:
        super().__init__()
        # Classical CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_conv = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        # Quanvolution (classical)
        self.qfilter = ClassicalQuanvolutionFilter()
        # Kernel mapping
        self.kernel_layer = RBFKernelLayer(input_dim=4 * 14 * 14, num_refs=64, gamma=gamma)
        # Self‑attention on CNN features
        self.attention = SelfAttentionBlock(embed_dim=64)
        # Classifier
        self.classifier = nn.Linear(4 * 14 * 14 + 64 + 64, num_classes)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Backbone
        conv_feat = self.features(x)
        conv_proj = self.fc_conv(conv_feat.view(bsz, -1))
        # Quanvolution
        qfilter_feat = self.qfilter(x)
        # Kernel features
        kernel_feat = self.kernel_layer(qfilter_feat)
        # Attention on backbone
        attn_feat = self.attention(conv_proj)
        # Concatenate all modalities
        combined = torch.cat([qfilter_feat, kernel_feat, attn_feat], dim=1)
        logits = self.classifier(combined)
        return self.norm(logits)

__all__ = ["HybridQuantumNat"]
