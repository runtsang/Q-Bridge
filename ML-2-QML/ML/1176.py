import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttentionEncoder(nn.Module):
    """
    Lightweight transformer encoder that aggregates spatial features.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, embed_dim)
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.ffn(x))
        return x

class HybridNet(nn.Module):
    """
    Classical backbone with convolutional layers followed by an attention encoder
    and a linear head. Returns a probability distribution over two classes.
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(0.2)
        self.drop2 = nn.Dropout2d(0.5)

        # Flattened feature map will be projected to embed_dim
        self.proj = nn.Linear(15 * 6 * 6, embed_dim)
        self.encoder = MultiHeadAttentionEncoder(embed_dim, num_heads)

        self.classifier = nn.Linear(embed_dim, 2)

        # Reproducibility
        torch.manual_seed(42)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = self.proj(x)
        # reshape to (seq_len=1, batch, embed_dim) for attention
        x = x.unsqueeze(0)
        x = self.encoder(x)
        x = x.squeeze(0)
        logits = self.classifier(x)
        probs = torch.softmax(logits, dim=-1)
        return probs

__all__ = ["HybridNet"]
