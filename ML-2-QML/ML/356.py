import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Simple residual block with two linear layers and layer norm."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        self.norm = nn.LayerNorm(out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.norm(out)
        if self.in_features!= self.out_features:
            x = nn.Linear(self.in_features, self.out_features)(x)
        return F.relu(out + x)

class AttentionLayer(nn.Module):
    """Scaled dot‑product self‑attention for 1‑D feature maps."""
    def __init__(self, feature_dim: int):
        super().__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key   = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = feature_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        return attn @ V

class QCNNModel(nn.Module):
    """Enhanced classical QCNN with residual blocks and self‑attention."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16, output_dim: int = 1):
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.res1 = ResidualBlock(hidden_dim, hidden_dim)
        self.attn = AttentionLayer(hidden_dim)
        self.pool1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh()
        )
        self.res2 = ResidualBlock(hidden_dim // 2, hidden_dim // 2)
        self.pool2 = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Tanh()
        )
        self.head = nn.Linear(hidden_dim // 4, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.res1(x)
        x = self.attn(x)
        x = self.pool1(x)
        x = self.res2(x)
        x = self.pool2(x)
        return torch.sigmoid(self.head(x))

def QCNN() -> QCNNModel:
    """Factory producing the enriched QCNN model."""
    return QCNNModel()
