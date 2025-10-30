import torch
from torch import nn
import torch.nn.functional as F

class TransformerBlockClassical(nn.Module):
    """
    Classical transformer block with multi‑head self‑attention and feed‑forward network.
    """
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
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_output))

class HybridEstimatorQNN(nn.Module):
    """
    A hybrid classical neural network that fuses feed‑forward regression, a QCNN‑style feature map,
    and a transformer encoder.  The architecture is fully differentiable and can be used for
    regression or classification depending on the ``classification`` flag.
    """
    def __init__(self,
                 input_dim: int = 2,
                 feature_dim: int = 8,
                 transformer_cfg: dict | None = None,
                 output_dim: int = 1,
                 classification: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        if transformer_cfg is None:
            transformer_cfg = dict(embed_dim=16, num_heads=4, num_blocks=2, ffn_dim=32)

        # QCNN‑style feature extractor
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, feature_dim),
            nn.Tanh()
        )

        # Transformer encoder
        self.transformer = nn.Sequential(*[
            TransformerBlockClassical(
                embed_dim=transformer_cfg["embed_dim"],
                num_heads=transformer_cfg["num_heads"],
                ffn_dim=transformer_cfg["ffn_dim"],
                dropout=dropout
            ) for _ in range(transformer_cfg["num_blocks"])
        ])

        # Classification head
        self.classification = classification
        self.classifier = nn.Linear(transformer_cfg["embed_dim"], output_dim)

        # Regularisation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = self.feature_map(x)

        # Transformer expects sequence dimension; reshape to (batch, seq_len=1, embed_dim)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)

        # Classification or regression
        x = self.dropout(x)
        x = self.classifier(x)

        if self.classification:
            return torch.sigmoid(x)
        return x

__all__ = ["HybridEstimatorQNN"]
