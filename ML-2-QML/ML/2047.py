import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumInspiredFeatureMap(nn.Module):
    """
    Classical approximation of a quantum feature map.
    Projects a scalar to a higher‑dimensional space using a
    trainable orthogonal basis followed by sinusoidal nonlinearity.
    """
    def __init__(self, in_dim: int = 1, out_dim: int = 8):
        super().__init__()
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        # Orthogonal initialization
        with torch.no_grad():
            w = torch.randn(out_dim, out_dim)
            q, _ = torch.linalg.qr(w)
            self.linear.weight.copy_(q.t())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_dim)
        z = self.linear(x)  # (batch, out_dim)
        return torch.sin(z) + torch.cos(z)

class QuantumHybridNet(nn.Module):
    """
    Classical CNN with a hybrid quantum‑inspired head.
    The network consists of a convolutional backbone, a sequence
    of fully connected layers, and two complementary heads:
    * a quantum‑inspired head that mimics a variational circuit
      via a feature map and a linear layer.
    * a plain classical head that directly processes the last
      fully connected output.
    The final probability is a blend of both heads.
    """
    def __init__(self):
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(15)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully connected stages
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum‑inspired head
        self.feature_map = QuantumInspiredFeatureMap(in_dim=1, out_dim=8)
        self.quantum_head = nn.Linear(8, 1)

        # Classical head
        self.classical_head = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop1(x)

        x = torch.flatten(x, 1)

        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(-1)  # (batch,)

        # Quantum‑inspired head
        q_feat = self.feature_map(x.unsqueeze(-1))  # (batch, 8)
        q_logits = self.quantum_head(q_feat).squeeze(-1)
        q_prob = torch.sigmoid(q_logits)

        # Classical head
        c_logits = self.classical_head(x.unsqueeze(-1)).squeeze(-1)
        c_prob = torch.sigmoid(c_logits)

        # Blend probabilities
        prob = 0.5 * q_prob + 0.5 * c_prob
        return torch.stack((prob, 1 - prob), dim=-1)

__all__ = ["QuantumHybridNet"]
