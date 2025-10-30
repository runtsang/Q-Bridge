"""Hybrid classical binary classifier with CNN + transformer encoder.

Combines the CNN feature extractor from ClassicalQuantumBinaryClassification.py
with a lightweight transformer encoder for richer representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlockClassical(nn.Module):
    """Simple transformer encoder block with multi‑head attention and FFN."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding compatible with transformer blocks."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) *
                              (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class HybridBinaryClassifier(nn.Module):
    """CNN + optional transformer encoder followed by a sigmoid head."""
    def __init__(self,
                 use_transformer: bool = True,
                 transformer_blocks: int = 1,
                 num_heads: int = 4,
                 ffn_dim: int = 128):
        super().__init__()
        # Feature extractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(0.2)
        self.drop2 = nn.Dropout2d(0.5)

        # Flatten size after convs (depends on input size; assumes 32x32)
        self.fc1 = nn.Linear(12 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Transformer encoder
        self.use_transformer = use_transformer
        if use_transformer:
            self.transformer = nn.Sequential(
                *[TransformerBlockClassical(84, num_heads, ffn_dim) for _ in range(transformer_blocks)]
            )
        else:
            self.transformer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if self.use_transformer:
            # transformer expects seq_len dimension
            x = x.unsqueeze(1)
            x = self.transformer(x)
            x = x.squeeze(1)

        probs = torch.sigmoid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridBinaryClassifier"]
