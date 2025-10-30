import torch
import torch.nn as nn
import torch.nn.functional as F

class Kernel(nn.Module):
    """Vectorised RBF kernel for CNN features."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (batch, dim), y: (num_support, dim)
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (batch, num_support, dim)
        return torch.exp(-self.gamma * diff.pow(2).sum(dim=-1))  # (batch, num_support)

class SamplerQNN(nn.Module):
    """Simple probabilistic sampler used as a classical surrogate for the quantum sampler."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

class HybridClassifier(nn.Module):
    """CNN‑based binary classifier that fuses a classical RBF kernel and a lightweight sampler."""
    def __init__(self, num_support: int = 10, gamma: float = 1.0):
        super().__init__()
        # Feature extractor
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1),
        )
        self.kernel = Kernel(gamma)
        # Learnable support vectors in feature space
        self.support_vectors = nn.Parameter(torch.randn(num_support, 1))
        # Linear head over kernel similarities
        self.fc = nn.Linear(num_support, 1)
        # Optional sampler for probabilistic outputs
        self.sampler = SamplerQNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.backbone(x).squeeze(-1)  # (batch, 1)
        # Compute kernel similarities to support vectors
        k = self.kernel(features, self.support_vectors.squeeze(-1))
        # Linear classification head
        logits = self.fc(k)
        probs = torch.sigmoid(logits)
        # Return two‑class logits
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridClassifier", "Kernel", "SamplerQNN"]
