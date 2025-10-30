import torch
from torch import nn
import numpy as np

#--- classical helpers -------------------------------------------------------
class ClassicalSelfAttention(nn.Module):
    """Attention block that emulates a quantum self‑attention style operation."""
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, rotation_params: torch.Tensor, entangle_params: torch.Tensor,
                inputs: torch.Tensor) -> torch.Tensor:
        query = inputs @ rotation_params.reshape(self.embed_dim, -1)
        key = inputs @ entangle_params.reshape(self.embed_dim, -1)
        scores = nn.functional.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs


class QCNNModel(nn.Module):
    """Classical surrogate for a quantum convolutional network."""
    def __init__(self):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


class QFCModel(nn.Module):
    """Quantum‑inspired fully‑connected block (4‑feature output)."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # dummy flatten to match the original shape
        flattened = x.view(bsz, -1)[:,:16*7*7]
        return self.norm(self.fc(flattened))


#--- main fraud‑detection network --------------------------------------------
class FraudDetectionNet(nn.Module):
    """
    End‑to‑end model that stitches together a classical convolutional
    core, an attention gate, and a quantum‑inspired fully‑connected layer.
    """
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.cnn = QCNNModel()
        self.attention = ClassicalSelfAttention(embed_dim=embed_dim)
        self.qfc = QFCModel()
        self.output = nn.Linear(4, 1)  # binary fraud score

    def forward(self, x: torch.Tensor,
                attn_rot: torch.Tensor = None,
                attn_ent: torch.Tensor = None) -> torch.Tensor:
        # 1. Convolutional feature extraction
        feat = self.cnn(x)

        # 2. Attention gating (requires rotation/entangle params; default random)
        if attn_rot is None:
            attn_rot = torch.randn(12, dtype=x.dtype, device=x.device)
        if attn_ent is None:
            attn_ent = torch.randn(12, dtype=x.dtype, device=x.device)

        gated = self.attention(attn_rot, attn_ent, feat)

        # 3. Quantum‑inspired fully‑connected projection
        qfc_out = self.qfc(gated)

        # 4. Final binary decision
        return torch.sigmoid(self.output(qfc_out))


__all__ = ["FraudDetectionNet"]
