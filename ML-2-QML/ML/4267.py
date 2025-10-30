import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalSelfAttention(nn.Module):
    """Simple dot‑product self‑attention with linear projections."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

class CNNFeatureExtractor(nn.Module):
    """Light‑weight CNN that produces a 4‑dimensional feature vector."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)

class HybridQLSTMTagger(nn.Module):
    """
    Classical hybrid tagger that optionally mixes a CNN feature extractor,
    a self‑attention block and a vanilla LSTM.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        use_self_attention: bool = False,
        use_cnn: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.use_cnn = use_cnn
        if use_cnn:
            self.cnn = CNNFeatureExtractor()
            self.cnn_proj = nn.Linear(4, embedding_dim)
        else:
            self.cnn = None
            self.cnn_proj = None

        self.use_self_attention = use_self_attention
        if use_self_attention:
            self.attention = ClassicalSelfAttention(embedding_dim)
        else:
            self.attention = None

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts either a sequence of token indices (LongTensor) or an image
        batch (FloatTensor with shape [B, C, H, W]).
        """
        if x.dim() == 4:  # image
            if self.cnn is None:
                raise ValueError("CNN feature extractor not enabled")
            features = self.cnn(x)  # shape [B, 4]
            embed = self.cnn_proj(features)  # [B, embed_dim]
            embed = embed.unsqueeze(1)  # [B, 1, embed_dim]
        else:  # sequence of token indices
            embed = self.embedding(x)  # [seq_len, batch, embed_dim]
            if self.use_self_attention:
                embed = self.attention.forward(embed)
        # LSTM expects [seq_len, batch, embed_dim] because batch_first=True
        lstm_out, _ = self.lstm(embed)
        logits = self.hidden2tag(lstm_out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQLSTMTagger"]
