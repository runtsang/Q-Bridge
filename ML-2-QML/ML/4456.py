import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class CNNFeatureExtractor(nn.Module):
    """Simple CNN for image feature extraction."""
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, embed_dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    """Classical transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class LSTMTagger(nn.Module):
    """Sequence tagging with a classical LSTM."""
    def __init__(self, embed_dim: int, hidden_dim: int, tagset_size: int):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        logits = self.hidden2tag(lstm_out)
        return F.log_softmax(logits, dim=-1)

class QTransformerTorchGen136(nn.Module):
    """Unified transformer model with optional CNN and LSTM branches."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_cnn: bool = False,
                 use_lstm: bool = False,
                 lstm_hidden_dim: int = 128,
                 **kwargs):
        super().__init__()
        self.use_cnn = use_cnn
        self.use_lstm = use_lstm
        if use_cnn:
            self.cnn = CNNFeatureExtractor(in_channels=3, embed_dim=embed_dim)
        else:
            self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        if use_lstm:
            self.lstm_tagger = LSTMTagger(embed_dim, lstm_hidden_dim, num_classes)
        else:
            self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cnn:
            x = self.cnn(x).unsqueeze(1)
        else:
            x = self.token_embed(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        if self.use_lstm:
            return self.lstm_tagger(x)
        else:
            x = x.mean(dim=1)
            return self.classifier(x)

__all__ = ["QTransformerTorchGen136"]
