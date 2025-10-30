import torch
from torch import nn
import torch.nn.functional as F

class QCNNHybridModel(nn.Module):
    """
    Classical network that extends the original QCNN by adding an
    attention mechanism over the outputs of the three convolutional
    stages and a residual skip connection.  The architecture remains
    lightweight to facilitate joint training with the quantum block.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 16, out_dim: int = 1) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attn = nn.Linear(hidden_dim, 3)  # attention over 3 conv outputs
        self.residual = nn.Identity()
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature_map(x)
        c1 = self.conv1(f)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        # stack conv outputs and apply attention
        cat = torch.stack([c1, c2, c3], dim=1)           # [B, 3, H]
        attn_weights = F.softmax(self.attn(cat), dim=1)  # [B, 3, H]
        out = torch.sum(attn_weights * cat, dim=1)       # [B, H]
        out = self.residual(out)
        return torch.sigmoid(self.head(out))

def QCNNHybrid() -> QCNNHybridModel:
    """
    Factory that returns a freshly‑initialized instance of
    :class:`QCNNHybridModel`.  No global state is modified.
    """
    return QCNNHybridModel()

__all__ = ["QCNNHybrid", "QCNNHybridModel"]
