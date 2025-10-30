import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Simple multi‑head self‑attention over flattened feature vectors."""
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.scale = 1 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, embed_dim)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each shape: (B, N, num_heads, head_dim)
        q = q.transpose(1, 2)  # (B, num_heads, N, head_dim)
        k = k.transpose(1, 2)  # (B, num_heads, N, head_dim)
        v = v.transpose(1, 2)  # (B, num_heads, N, head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, v)  # (B, num_heads, N, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, self.embed_dim)
        return self.out_proj(attn_output)

class QuantumHybridClassifier(nn.Module):
    """Classical CNN + self‑attention head for binary classification."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.attn = MultiHeadAttention(embed_dim=120, num_heads=4)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(1)  # shape (B, 1, 120) for attention
        x = self.attn(x)
        x = x.squeeze(1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = torch.sigmoid(x)
        return torch.cat((probs, 1 - probs), dim=-1)
