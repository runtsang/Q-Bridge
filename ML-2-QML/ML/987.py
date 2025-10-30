"""HybridQuantumBinaryClassifier – Classical transformer head for binary classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    """A single transformer encoder layer."""
    def __init__(self, embed_dim: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class HybridQuantumBinaryClassifier(nn.Module):
    """Classical CNN + transformer head for binary classification."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        self.fc1 = nn.Linear(55815, 120)
        self.transformer = TransformerBlock(embed_dim=12, num_heads=2)
        self.seq_len = 10
        self.embed_dim = 12
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = x.view(x.size(0), self.seq_len, self.embed_dim)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        logits = self.fc3(x)
        probs = torch.sigmoid(logits)
        return torch.cat((probs.unsqueeze(-1), (1 - probs).unsqueeze(-1)), dim=-1)

__all__ = ["HybridQuantumBinaryClassifier"]
