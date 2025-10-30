import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# Classical Quanvolution filter – 2×2 patches mapped to a 4‑dim output
# ------------------------------------------------------------------
class QuanvolutionFilter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)


# ------------------------------------------------------------------
# Classical transformer block (multi‑head attention + FFN)
# ------------------------------------------------------------------
class TransformerBlockClassical(nn.Module):
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


# ------------------------------------------------------------------
# Main hybrid model – classical variant
# ------------------------------------------------------------------
class HybridNATModel(nn.Module):
    def __init__(
        self,
        use_quanvolution: bool = True,
        num_classes: int = 10,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Feature extractor
        if use_quanvolution:
            self.feature_extractor = QuanvolutionFilter()
            self.feature_dim = 4 * 14 * 14
        else:
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.feature_dim = 16 * 7 * 7

        # Projection to transformer space
        self.fc = nn.Linear(self.feature_dim, embed_dim)
        self.norm = nn.BatchNorm1d(embed_dim)

        # Transformer encoder
        self.transformer = nn.Sequential(
            *[
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )

        # Classifier
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        if isinstance(features, torch.Tensor):
            features = features.view(features.size(0), -1)
        x = self.fc(features)
        x = self.norm(x)
        x = x.unsqueeze(1)  # sequence length = 1
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


__all__ = ["HybridNATModel"]
