import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 patch convolution to produce 4 features per patch."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)  # shape [B,4,14,14]
        return features.view(x.size(0), -1)  # [B, 784]

class QCNNBlock(nn.Module):
    """QCNN‑inspired feature extractor using linear layers and tanh activations."""
    def __init__(self, in_features: int, hidden: int = 128) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(in_features, hidden), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(hidden, hidden - 16), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(hidden - 16, hidden // 2), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(hidden // 2, hidden // 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(hidden // 4, hidden // 4), nn.Tanh())
        self.head = nn.Linear(hidden // 4, hidden // 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return self.head(x)

class TransformerBlockClassical(nn.Module):
    """Standard multi‑head attention with feed‑forward network."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_output))

class HybridQuanvolutionModel(nn.Module):
    """Hybrid model combining quanvolution, QCNN, and transformer."""
    def __init__(self,
                 num_classes: int = 10,
                 use_quantum: bool = False,
                 transformer_layers: int = 2,
                 embed_dim: int = 64,
                 num_heads: int = 4,
                 ffn_dim: int = 128):
        super().__init__()
        self.use_quantum = use_quantum
        self.qfilter = QuanvolutionFilter()
        self.qcnn = QCNNBlock(in_features=4 * 14 * 14, hidden=embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim) for _ in range(transformer_layers)]
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        features = self.qcnn(features)
        features = features.unsqueeze(1)  # [B,1,embed_dim]
        x = self.transformers(features)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionModel"]
