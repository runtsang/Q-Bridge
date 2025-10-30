import torch
from torch import nn

class QCNNPlus(nn.Module):
    """
    Classical QCNN analogue with attention-based feature fusion.
    Mirrors the quantum architecture but adds dynamic weighting of intermediate
    representations to improve expressiveness.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 16):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())

        self.conv1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh())

        self.conv2 = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim // 2), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.Tanh())

        self.conv3 = nn.Sequential(nn.Linear(hidden_dim // 4, hidden_dim // 4), nn.Tanh())

        # Attention weights over concatenated intermediate features
        num_features = hidden_dim + hidden_dim // 2 + hidden_dim // 4 + hidden_dim // 4
        self.attn_weights = nn.Parameter(torch.ones(num_features))

        self.head = nn.Linear(hidden_dim // 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_map = self.feature_map(x)       # [B, hidden_dim]
        c1 = self.conv1(f_map)            # [B, hidden_dim]
        p1 = self.pool1(c1)               # [B, hidden_dim//2]
        c2 = self.conv2(p1)               # [B, hidden_dim//2]
        p2 = self.pool2(c2)               # [B, hidden_dim//4]
        c3 = self.conv3(p2)               # [B, hidden_dim//4]

        concat = torch.cat([f_map, p1, p2, c3], dim=1)  # [B, num_features]

        # Apply attention weights elementwise then sum
        attn = torch.sigmoid(self.attn_weights)  # [num_features]
        fused = (concat * attn).sum(dim=1, keepdim=True)  # [B, 1]

        logits = self.head(fused)  # [B, 1]
        return torch.sigmoid(logits)
