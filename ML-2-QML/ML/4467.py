import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class ConvFilter(nn.Module):
    """Lightweight 2‑D convolution emulating a quantum filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

class ClassicalSelfAttention(nn.Module):
    """Dot‑product self‑attention with learnable projections."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.weight_q = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.weight_k = nn.Parameter(torch.randn(embed_dim, embed_dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = torch.matmul(x, self.weight_q)
        key = torch.matmul(x, self.weight_k)
        scores = F.softmax(query @ key.transpose(-2, -1) / math.sqrt(self.embed_dim), dim=-1)
        return scores @ x

class TransformerBlock(nn.Module):
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
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class TextClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_blocks: int,
                 ffn_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
                                      for _ in range(num_blocks)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_enc(x)
        x = self.blocks(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

class UnifiedHybridClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, use_transformer: bool = True):
        super().__init__()
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
            nn.Flatten(),
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1)
        )
        self.image_self_attn = ClassicalSelfAttention(embed_dim=1)
        self.image_conv_filter = ConvFilter(kernel_size=2, threshold=0.0)
        self.use_transformer = use_transformer
        if use_transformer:
            self.text_branch = TextClassifier(
                vocab_size=30522, embed_dim=64, num_heads=4,
                num_blocks=2, ffn_dim=256, num_classes=1, dropout=0.1
            )
        else:
            self.text_branch = nn.Sequential(
                nn.Embedding(30522, 64),
                nn.Flatten(),
                nn.Linear(64 * 10, 1)
            )
        self.classifier = nn.Linear(2, num_classes if num_classes > 2 else 1)
    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_branch(image)
        img_feat = self.image_self_attn(img_feat.unsqueeze(1)).squeeze(1)
        img_feat = self.image_conv_filter(img_feat.view(-1, 1, 1, 1))
        txt_feat = self.text_branch(text)
        combined = torch.cat([img_feat, txt_feat], dim=-1)
        out = self.classifier(combined)
        return torch.cat([out, 1 - out], dim=-1)
