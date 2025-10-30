import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBranch(nn.Module):
    """Feature‑wise attention that re‑weights the flattened CNN activations."""
    def __init__(self, in_features: int, hidden_dim: int = 64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attn(x)

class ClassicalHybridFunction(nn.Module):
    """Simple sigmoid head with optional bias shift."""
    def __init__(self, shift: float = 0.0):
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x + self.shift)

class HybridClassifier(nn.Module):
    """Classical CNN with an optional attention branch and sigmoid head."""
    def __init__(self, use_attention: bool = True, shift: float = 0.0):
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Fully‑connected head
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Optional modules
        self.use_attention = use_attention
        if use_attention:
            self.attn_branch = AttentionBranch(84)
        self.head = ClassicalHybridFunction(shift)

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
        if self.use_attention:
            x = self.attn_branch(x)
        x = self.fc3(x)
        probs = self.head(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridClassifier"]
