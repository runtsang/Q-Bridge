import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """Additive self‑attention over a batch of feature vectors."""
    def __init__(self, in_dim: int, attn_dim: int):
        super().__init__()
        self.query = nn.Linear(in_dim, attn_dim)
        self.key   = nn.Linear(in_dim, attn_dim)
        self.value = nn.Linear(in_dim, attn_dim)
        self.scale = math.sqrt(attn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_dim)
        q = self.query(x)   # (batch, attn_dim)
        k = self.key(x)     # (batch, attn_dim)
        scores = torch.matmul(q, k.t()) / self.scale  # (batch, batch)
        attn = torch.softmax(scores, dim=-1)          # (batch, batch)
        out = torch.matmul(attn, self.value(x))       # (batch, attn_dim)
        return out

class HybridDenseHead(nn.Module):
    """Classical dense head that applies dropout and a sigmoid."""
    def __init__(self, in_features: int, dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.sigmoid(x)

class QCNet(nn.Module):
    """CNN backbone with optional attention and a classical head."""
    def __init__(self, use_attention: bool = True, attn_dim: int = 84):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.use_attention = use_attention
        if use_attention:
            self.attn = SelfAttention(120, attn_dim)
        else:
            self.attn = None

        self.head = HybridDenseHead(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        if self.attn is not None:
            x = self.attn(x)
        x = self.fc3(x)  # (batch, 1)
        probs = self.head(x)  # (batch, 1)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["SelfAttention", "HybridDenseHead", "QCNet"]
