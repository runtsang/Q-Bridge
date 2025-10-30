import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
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

class HybridNAT(nn.Module):
    """
    Classical hybrid model that combines a CNN feature extractor,
    a transformer encoder (with optional quantum submodules), and a
    final linear classifier/regressor.
    """
    def __init__(
        self,
        num_classes: int = 1,
        embed_dim: int = 64,
        num_heads: int = 8,
        num_blocks: int = 4,
        ffn_dim: int = 256,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.patch_size = patch_size
        self.project = nn.Linear(32 * patch_size * patch_size, embed_dim)
        self.pos_enc = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ffn_dim, dropout=0.1) for _ in range(num_blocks)]
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Logits (B, num_classes) or regression output (B, 1).
        """
        features = self.cnn(x)
        # unfold feature map into patches
        b, c, h, w = features.shape
        patches = features.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(b, c, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(b, -1, c * self.patch_size * self.patch_size)
        seq = self.project(patches)
        seq = self.pos_enc(seq)
        seq = self.transformer(seq)
        pooled = seq.mean(dim=1)
        return self.classifier(pooled)
