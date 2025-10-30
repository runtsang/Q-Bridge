import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer input."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class HybridQuantumNAT(nn.Module):
    """Hybrid classical‑quantum feature extractor with attention."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        feat_dim = 32 * 4 * 4
        self.attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feats = self.features(x)
        feats = feats.view(bsz, -1)          # (batch, feat_dim)
        feats_seq = feats.unsqueeze(1)       # (batch, 1, feat_dim)
        attn_out, _ = self.attn(feats_seq, feats_seq, feats_seq)  # (batch, 1, feat_dim)
        return attn_out.squeeze(1)           # (batch, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embed(x)
        out = self.fc(embedded)
        return self.norm(out)

__all__ = ["HybridQuantumNAT"]
