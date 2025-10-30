import math
import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class TransformerBlockClassical(nn.Module):
    """A classical transformer encoder block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim)
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

class UnifiedQuantumTransformLayer(nn.Module):
    """
    Classical implementation that merges a fully‑connected mapping with
    a transformer encoder. The API mirrors the original FCL.run and
    TextClassifier.forward, enabling drop‑in replacement.
    """
    def __init__(self,
                 n_features: int = 1,
                 embed_dim: int = 32,
                 num_heads: int = 4,
                 ffn_dim: int = 64,
                 num_blocks: int = 2,
                 dropout: float = 0.1,
                 device: str = "cpu"):
        super().__init__()
        self.device = device
        self.linear = nn.Linear(n_features, embed_dim, bias=True)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
             for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)

    def run(self, thetas: torch.Tensor) -> torch.Tensor:
        """
        Mimic the classical FCL.run: apply linear mapping,
        take mean over the sequence, and return a scalar.
        """
        x = self.linear(thetas)
        x = torch.tanh(x).mean(dim=0, keepdim=True)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer encoder.
        """
        x = self.linear(x)
        x = self.pos_encoder(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)
