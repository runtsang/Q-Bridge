import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --------------------------------------------------------------------------- #
#  Classical CNN backbone
# --------------------------------------------------------------------------- #
class _CNNBackbone(nn.Module):
    """2‑layer ConvNet mirroring the original QFCModel architecture."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

# --------------------------------------------------------------------------- #
#  Classical self‑attention helper
# --------------------------------------------------------------------------- #
class SelfAttention(nn.Module):
    """Simple self‑attention block using linear projections."""
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        scores = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, v)

# --------------------------------------------------------------------------- #
#  Hybrid model
# --------------------------------------------------------------------------- #
class QuantumNAT__gen318(nn.Module):
    """Hybrid CNN‑Quantum‑Attention model for image classification."""
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _CNNBackbone()
        self.fc = nn.Linear(16 * 7 * 7, 64)
        self.attn = SelfAttention(embed_dim=64)
        self.out = nn.Linear(64, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        flattened = features.view(x.size(0), -1)
        fc_out = F.relu(self.fc(flattened))
        attn_out = self.attn(fc_out, fc_out, fc_out)
        logits = self.out(attn_out)
        return self.norm(logits)

__all__ = ["QuantumNAT__gen318"]
