import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
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
            nn.Linear(ffn_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class QuantumHybridNAT(nn.Module):
    """
    Classical implementation of the hybrid Quantum‑NAT architecture.
    Supports CNN‑backbone or transformer encoder.
    """
    def __init__(
        self,
        mode: str = 'cnn',
        vocab_size: int = 30522,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 128,
        num_classes: int = 4,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.device = device or torch.device('cpu')

        # CNN backbone
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
            nn.Linear(64, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

        # Transformer encoder
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlockClassical(embed_dim, num_heads, ffn_dim) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'cnn':
            bsz = x.shape[0]
            feat = self.features(x)
            flat = feat.view(bsz, -1)
            out = self.fc(flat)
            return self.norm(out)
        else:
            tokens = self.token_embedding(x)
            tokens = self.pos_encoder(tokens)
            for blk in self.transformer_blocks:
                tokens = blk(tokens)
            x = tokens.mean(dim=1)
            return self.dropout(self.classifier(x))

__all__ = ["QuantumHybridNAT"]
