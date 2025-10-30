import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuanvolutionFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv(x)
        return features.view(x.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)

class RBFKernel(nn.Module):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class KernelClassifier(nn.Module):
    def __init__(self, prototypes: np.ndarray, num_classes: int, gamma: float = 1.0):
        super().__init__()
        self.prototypes = nn.Parameter(torch.tensor(prototypes, dtype=torch.float32))
        self.gamma = gamma
        self.linear = nn.Linear(prototypes.shape[0], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = torch.exp(-self.gamma * torch.cdist(x, self.prototypes)**2)
        logits = self.linear(k)
        return F.log_softmax(logits, dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

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

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

class QTransformerTorch(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_quanvolution: bool = False,
                 use_lstm: bool = False,
                 use_kernel: bool = False,
                 kernel_gamma: float = 1.0,
                 prototypes: np.ndarray | None = None):
        super().__init__()
        self.use_quanvolution = use_quanvolution
        self.use_lstm = use_lstm
        self.use_kernel = use_kernel

        if self.use_quanvolution:
            self.qfilter = QuanvolutionFilter()
            in_features = 4 * 14 * 14
        else:
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)
            self.pos_embedding = PositionalEncoder(embed_dim)
            in_features = embed_dim

        if self.use_lstm:
            self.lstm = nn.LSTM(in_features, embed_dim, batch_first=True)
        else:
            self.lstm = None

        self.transformers = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )

        if self.use_kernel:
            if prototypes is None:
                raise ValueError("prototypes must be provided when use_kernel=True")
            self.classifier = KernelClassifier(prototypes, num_classes, gamma=kernel_gamma)
        else:
            self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quanvolution:
            features = self.qfilter(x)
            x = features
        else:
            tokens = self.token_embedding(x)
            x = self.pos_embedding(tokens)

        if self.use_lstm:
            x, _ = self.lstm(x)

        x = self.transformers(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits

__all__ = [
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "RBFKernel",
    "KernelClassifier",
    "MultiHeadAttention",
    "FeedForward",
    "TransformerBlock",
    "PositionalEncoder",
    "LSTMTagger",
    "QTransformerTorch",
]
