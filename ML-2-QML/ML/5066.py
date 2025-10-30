"""Hybrid LSTMTagger with classical LSTM, convolutional feature extraction, and self‑attention.

The module combines a classical LSTM with a lightweight 2×2 convolution filter
followed by a self‑attention layer.  It preserves the public interface of the
original QLSTM example so it can be dropped in as a drop‑in replacement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalConvFilter(nn.Module):
    """Simple 2×2 stride‑2 convolution that mimics the behaviour of the
    original quanvolution filter but stays completely classical."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=1,
            kernel_size=kernel_size, stride=kernel_size, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1, H, W]
        features = self.conv(x)
        # flatten to a single feature vector per example
        return features.view(x.size(0), -1)

class ClassicalSelfAttention(nn.Module):
    """Scaled dot‑product self‑attention implemented with linear layers."""
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len, batch, embed_dim]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # transpose to [batch, seq_len, embed_dim]
        q_t = q.transpose(0, 1)
        k_t = k.transpose(0, 1)
        v_t = v.transpose(0, 1)
        scores = torch.softmax(
            q_t @ k_t.transpose(-2, -1) / torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32)),
            dim=-1
        )
        out = scores @ v_t
        return out.transpose(0, 1)  # back to [seq_len, batch, embed_dim]

class LSTMTagger(nn.Module):
    """Hybrid sequence‑tagger that replaces the quantum LSTM with a classical one
    while keeping the rest of the architecture identical."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        conv_kernel: int = 2,
        conv_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.conv = ClassicalConvFilter(kernel_size=conv_kernel, threshold=conv_threshold)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.attention = ClassicalSelfAttention(hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        # sentence: [seq_len, batch, 1, H, W]
        embeds = self.word_embeddings(sentence)
        seq_len, batch, *_ = embeds.shape
        features = []
        for i in range(seq_len):
            img = sentence[i]  # [batch, 1, H, W]
            feat = self.conv(img)
            features.append(feat)
        conv_out = torch.stack(features, dim=0)  # [seq_len, batch, conv_dim]
        lstm_out, _ = self.lstm(conv_out)
        att_out = self.attention(lstm_out)
        logits = self.hidden2tag(att_out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["LSTMTagger"]
