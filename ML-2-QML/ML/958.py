import torch
from torch import nn

class QCNN(nn.Module):
    """Extended QCNN architecture with residual connections and dropout."""
    def __init__(self, input_dim: int = 8, hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh()
        )
        self.conv1 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh()
        )
        self.pool1 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.Tanh()
        )
        self.pool2 = nn.Sequential(
            nn.Linear(hidden_dims[3], hidden_dims[4]),
            nn.Tanh()
        )
        self.conv3 = nn.Sequential(
            nn.Linear(hidden_dims[4], hidden_dims[5]),
            nn.Tanh()
        )
        self.dropout = nn.Dropout(p=0.2)
        self.head = nn.Linear(hidden_dims[5], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.feature_map(x)
        out = self.conv1(out) + out
        out = self.pool1(out) + out
        out = self.conv2(out) + out
        out = self.pool2(out) + out
        out = self.conv3(out) + out
        out = self.dropout(out)
        out = self.head(out)
        return torch.sigmoid(out)

def QCNNFactory() -> QCNN:
    """Return an instantiated QCNN model."""
    return QCNN()

__all__ = ["QCNN", "QCNNFactory"]
