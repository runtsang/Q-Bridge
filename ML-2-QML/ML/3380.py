import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridSelfAttention(nn.Module):
    """Hybrid CNN + self‑attention model with learnable rotation and entangle parameters."""
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Convolutional backbone (similar to QFCModel)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.embed_dim)
        )
        
        # Learnable attention parameters
        self.rotation_params = nn.Parameter(torch.randn(self.embed_dim, self.embed_dim))
        self.entangle_params = nn.Parameter(torch.randn(self.embed_dim, self.embed_dim))
        self.norm = nn.BatchNorm1d(self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        features = self.features(x)          # shape: [B, 16, 7, 7]
        flattened = features.view(features.size(0), -1)
        proj = self.fc(flattened)            # shape: [B, embed_dim]
        
        # Classical self‑attention
        query = torch.matmul(proj, self.rotation_params)          # [B, embed_dim]
        key   = torch.matmul(proj, self.entangle_params)          # [B, embed_dim]
        scores = F.softmax(torch.matmul(query, key.t()) / np.sqrt(self.embed_dim), dim=-1)
        attn_out = torch.matmul(scores, proj)                     # [B, embed_dim]
        
        return self.norm(attn_out)

__all__ = ["HybridSelfAttention"]
