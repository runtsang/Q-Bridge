import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 500):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class TransformerBlockClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class FCL(nn.Module):
    """Classical surrogate for a fully connected quantum layer."""
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simulate quantum expectation via tanh and mean over batch
        return torch.tanh(self.linear(x)).mean(dim=1, keepdim=True)

class HybridNet(nn.Module):
    """Hybrid CNN + optional transformer + quantum surrogate classifier."""
    def __init__(
        self,
        num_classes: int = 1,
        use_transformer: bool = False,
        transformer_layers: int = 2,
        embed_dim: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 128,
    ) -> None:
        super().__init__()
        # ConvNet backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(32, embed_dim)
        self.use_transformer = use_transformer
        if use_transformer:
            self.pos_enc = PositionalEncoder(embed_dim)
            self.transformers = nn.Sequential(
                *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim) for _ in range(transformer_layers)]
            )
        self.fcl = FCL(embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        if self.use_transformer:
            seq = x.unsqueeze(1)          # [B, 1, E]
            seq = self.pos_enc(seq)
            seq = self.transformers(seq)
            x = seq.squeeze(1)             # [B, E]
        logits = self.fcl(x)                # [B, 1]
        probs = torch.sigmoid(logits).squeeze(-1)  # [B]
        return torch.stack([probs, 1 - probs], dim=-1)

__all__ = ["HybridNet"]
